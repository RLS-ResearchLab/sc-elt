from utils import (maybe_resume,
                   InfiniteDataLoader,
                   move_to_device,
                   save_checkpoint,set_seed,
                   setup_environment,build_accelerator,
                   build_logger,
                   build_optimizer,
                   build_scheduler,
                   load_config,
                   freeze_model,
                   build_vae_from_checkpoint,
                   ELTSchedule,
                   sample_intermediate_loops,
                   build_distillation,
                   build_repa,
                   denormalize,
                   EMA
                   )
from eval import extract_images,visualize_reconstruction,build_evaluators,build_fid_metric
from model import build_model
from  diffusion  import  build_diffusion
from sample import build_sampler
from losses import build_loss
from  data import build_dataloader,extract_labels,compute_scaling_factor
import  torch
import wandb,os
import copy

# ============================================================
# Base Trainer
# ============================================================
class BaseTrainer:

    def __init__(self,cfg,model,optimizer,criterion,train_loader,accelerator,
        device,checkpoint_dir,scheduler=None,ema=None,logger=None,evaluators=None):

        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.accelerator = accelerator
        self.device = device
        self.scheduler = scheduler
        self.ema = ema
        self.logger = logger
        self.evaluators = (evaluators if evaluators is not None else {})
        self.checkpoint_dir = checkpoint_dir

    # Interface
    def setup(self):
        """
        Optional initialization before training.
        """
        pass

    def train(self):
        raise NotImplementedError

    def train_step(self, batch,step=None):
        raise NotImplementedError

    def validate(self,step):
        raise NotImplementedError

    def log(self,step,train_output):
        raise NotImplementedError

    def save_checkpoint(self,step):
        raise NotImplementedError

    def save_final_checkpoint(self):
        raise NotImplementedError
    
    def _log_evaluation(self, step, name, result):
        metrics = result.get("metrics", {})
        logs = {**{f"eval/{name}/{k}": v for k, v in metrics.items()},"step": step,}
        images = result.get("images", None)
        if images is not None:
            grid = visualize_reconstruction(images["originals"],images["reconstructions"])
            logs[f"eval/{name}/reconstruction"] = wandb.Image(grid)
        if self.accelerator.is_main_process:
            wandb.log(logs)
            print(f"eval step {step} | {name}: "+" ".join(f"{k}: {v:.5f}"for k, v in metrics.items()))
    def _use_ema(self):
            if self.ema is not None:
                return self.ema.apply_shadow(self.raw_model)
            return None

    def _restore_ema(self, backup):
        if backup is None:
            return
        if self.ema is not None:
            self.ema.restore(self.raw_model,backup,)
            
    def reset_running_losses(self):

        self.running_sums = {}
        self.running_count = 0

    def update_running_losses(self, losses, batch_size):
        for name, value in losses.items():
            if name not in self.running_sums:
                self.running_sums[name] = 0.0
            self.running_sums[name] += (value.detach().item() * batch_size)
        self.running_count += batch_size

    def get_running_metrics(self):
        if self.running_count == 0:
            return {}
        return {
            name: total / self.running_count
            for name, total in self.running_sums.items()
        }
    
    def run_evaluation(self, step):
        backup = self._use_ema()
        results = {}

        try:
            self.raw_model.eval()
            for name,evaluator in self.evaluators.items():
                result = evaluator.evaluate()
                self._log_evaluation(step,name,result,)
                results[name] = result
        finally:
            self._restore_ema(backup)
            self.raw_model.train()
        return results

            
class VAETrainer(BaseTrainer):
    def __init__(self, cfg, model, optimizer, criterion, train_loader, accelerator, device, checkpoint_dir, scheduler=None, ema=None, logger=None, evaluators=None):
        super().__init__(cfg, model, optimizer, criterion, train_loader, accelerator, device, checkpoint_dir, scheduler, ema, logger, evaluators)
    def setup(self):
        self.model, self.optimizer,self.scheduler, self.train_loader = (
            self.accelerator.prepare(
                self.model,
                self.optimizer,
                self.scheduler,
                self.train_loader,
                )
            )
        self.best_metric = float("inf") 
        self.raw_model = self.accelerator.unwrap_model(self.model)

        if self.cfg["train"]["use_compile"]:
            self.model = torch.compile(self.model)
        self.start_step  = maybe_resume(cfg=self.cfg,
                                         model=self.raw_model,
                                         optimizer=self.optimizer,
                                         scheduler=self.scheduler,
                                         ema=self.ema,
                                         device=self.device)
                
                    

        self.total_steps =  self.cfg["train"]["total_steps"]
        self.train_iter = InfiniteDataLoader(self.train_loader)
        self.running_sums = {}
        self.running_count = 0
        
    def train_step(self, batch,step=None):
        images = extract_images(batch)
        with self.accelerator.accumulate(self.model):
            with self.accelerator.autocast():
                out = self.model(images)
                losses = self.criterion(out["recon"],images,out["mu"],out["logvar"])
                loss = losses["loss"]
            self.optimizer.zero_grad(set_to_none=True)
            self.accelerator.backward(loss)
            self.accelerator.clip_grad_norm_(self.model.parameters(),self.cfg["train"]["grad_clip_norm"])
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
        if self.ema is not None:
            self.ema.update(self.raw_model,step =step)
        return {
            "losses": losses,
            "batch_size": images.size(0)
            }

    def log(self, step, train_output):
        self.update_running_losses(train_output["losses"], train_output["batch_size"])
        if step % self.cfg["train"]["log_every"] != 0:
            return
        metrics = self.get_running_metrics()
        if self.accelerator.is_main_process:
            current_lr = self.scheduler.get_last_lr()[0] if self.scheduler is not None else self.optimizer.param_groups[0]["lr"]
            wandb.log({**{f"train/{k}": v for k, v in metrics.items()}, "train/lr": current_lr, "step": step})
            if step % 10000 == 0:
                print(f"step {step} | lr: {current_lr:.6e}")
            print(f"step {step} | " + " ".join(f"{k}: {v:.5f}" for k, v in metrics.items()))
        self.reset_running_losses()

    def validate(self, step):
        if step == 0:
            return
        if step % self.cfg["eval"]["every"] != 0:
            return
        if not self.accelerator.is_main_process:
            return
        results = self.run_evaluation(step)
        current = results["reconstruction"]["metrics"]["reconstruction_mse"]
        if current < self.best_metric:
            self.best_metric = current
            self.best_step = step
            # track the state, but don't save to disk yet
            self.best_state = {
                "model": copy.deepcopy(self.raw_model.state_dict()),
                "ema": copy.deepcopy(self.ema.shadow) if self.ema is not None else None,
            }
            print(f"step {step}: new best reconstruction_mse={current:.6f} (will save at end of training)")

    def save_checkpoint(self, step):

        if step % self.cfg["train"]["ckpt_every"] != 0:
            return
        if not self.accelerator.is_main_process:
            return
        path = os.path.join(self.checkpoint_dir, f"{self.cfg['model']['name']}_run{self.cfg['run_number']}_{step}.pt")
        save_checkpoint(path=path,model=self.raw_model,optimizer=self.optimizer,ema=self.ema,epoch=step,cfg=self.cfg,scheduler=self.scheduler)
    def save_final_checkpoint(self):

        if not self.accelerator.is_main_process:
            return
        path = os.path.join(self.checkpoint_dir, f"{self.cfg['model']['name']}_run{self.cfg['run_number']}_final.pt")
        save_checkpoint(path=path, model=self.raw_model, optimizer=self.optimizer, ema=self.ema, epoch=self.total_steps, cfg=self.cfg,scheduler=self.scheduler)

    def save_best_checkpoint(self):
        if not self.accelerator.is_main_process:
            return
        if not hasattr(self, "best_state"):
            return
        path = os.path.join(self.checkpoint_dir, f"{self.cfg['model']['name']}_run{self.cfg['run_number']}_best.pt")
        # temporarily load best weights into raw_model, save, then restore current weights
        current_state = copy.deepcopy(self.raw_model.state_dict())
        self.raw_model.load_state_dict(self.best_state["model"])
        save_checkpoint(
            path=path, model=self.raw_model, optimizer=self.optimizer,
            scheduler=self.scheduler, ema=self.ema, epoch=self.best_step, cfg=self.cfg
        )
        self.raw_model.load_state_dict(current_state)
    def train(self):
        self.setup()
        for step in range(self.start_step, self.total_steps):
            batch = next(self.train_iter)
            train_output = self.train_step(batch,step)
            self.log(step, train_output)
            self.validate(step)
            self.save_checkpoint(step)
        self.save_final_checkpoint()
        self.save_best_checkpoint()

def build_vae_trainer(cfg):
    set_seed(cfg["seed"])
    checkpoint_dir = setup_environment(cfg)
    accelerator = build_accelerator(cfg)
    device = accelerator.device
    logger = build_logger(cfg,accelerator)
    train_loader = build_dataloader(cfg["data"], split="train")
    test_loader = build_dataloader(cfg["data"],split="test")
    loaders = {
        "train": train_loader,
        "test": test_loader,
    }
    model = build_model(cfg)
    criterion = build_loss(cfg)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg) 
    window = cfg["train"]["ema_window_ratio"] * cfg["train"]["total_steps"]
    ema_decay = 1 - 1 / window
    ema = EMA(model, decay=ema_decay)
    evaluators = build_evaluators(cfg, model, loaders,device)  # <- needs loaders/fid_metric, see below
    return VAETrainer(
        cfg, model, optimizer, criterion, train_loader, accelerator, device,
        checkpoint_dir, scheduler=scheduler, ema=ema,logger=logger, evaluators=evaluators,
    )

# DIT TRAINER
class DiTTrainer(BaseTrainer):
    def __init__(self, cfg, model,vae,diffusion, optimizer, criterion, train_loader, accelerator, device, checkpoint_dir,scaling_factor,repa = None,repa_encoder = None, scheduler=None, ema=None, logger=None, evaluators=None,distill=None):
        super().__init__(cfg, model, optimizer, criterion, train_loader, accelerator, device, checkpoint_dir, scheduler, ema, logger, evaluators)

        self.use_elt = cfg["elt"]["enabled"]

        if self.use_elt and cfg["model"]["name"] != "looped_dit":
            raise ValueError(
                f"elt.enabled=True requires model.name='looped_dit', "
                f"got '{cfg['model']['name']}'"
            )
        if self.use_elt and distill is None:
            raise ValueError(
                "elt.enabled=True requires a distill loss function to be provided "
                "(e.g. distill=F.mse_loss) -- use elt.enabled=False for a looped "
                "model with no distillation instead of passing distill=None"
            )
        self.vae = vae
        self.scaling_factor = scaling_factor
        self.repa = repa
        self.repa_encoder = repa_encoder
        self.diffusion = diffusion
        self.distill = distill
        self.distill_scheduler = None
        if cfg["elt"]["enabled"]:
            self.distill_scheduler = ELTSchedule(cfg["elt"]["distillation"],cfg["train"]["total_steps"])

    def setup(self):
        self.model, self.optimizer,self.scheduler, self.train_loader = (
                    self.accelerator.prepare(
                        self.model,
                        self.optimizer,
                        self.scheduler,
                        self.train_loader,
                        )
                    )
        self.best_metric = float("inf")
        freeze_model(self.vae)
        if self.repa_encoder is not None:
            freeze_model(self.repa_encoder)
            self.repa.to(self.device)
            self.repa_encoder.eval()
        self.vae.to(self.device)
        self.vae.eval()

        self.diffusion.to(self.device)
        self.diffusion.eval()
        self.raw_model = self.accelerator.unwrap_model(self.model)
        if self.cfg["train"]["use_compile"]:
            self.model = torch.compile(self.model)
        self.start_step = maybe_resume(cfg=self.cfg, model=self.raw_model,
                                     optimizer=self.optimizer,scheduler=self.scheduler, ema=self.ema, device=self.device)

        self.total_steps =  self.cfg["train"]["total_steps"]
        self.train_iter = InfiniteDataLoader(self.train_loader)
        self.running_sums = {}
        self.running_count = 0

    def _parse_model_output(self, output):
        if isinstance(output, tuple) and len(output) == 3:
            pred, c, history = output
            return {
                "pred": pred,
                "c": c,
                "history": history,
            }

        return {
            "pred": output,
            "c": None,
            "history": None,
        }
    def train_step(self,batch,step = None):
        images = extract_images(batch)
        labels = extract_labels(batch) if self.cfg["conditioning"]["enabled"] else None
        with torch.no_grad():
            mu, logvar = self.vae.encoder(images)
            latents = mu*self.scaling_factor
            
        with self.accelerator.accumulate(self.model):
            with self.accelerator.autocast():
                x_t,t,noise = self.diffusion(latents)
                elt_loss = None
                if  self.use_elt:
                    record = sample_intermediate_loops(self.cfg, self.raw_model.loop_cfg.loop_steps)
                    output = self.model(x_t,t,labels,record=record)
                else:
                    output = self.model(x_t,t,labels)

                model_output = self._parse_model_output(output)
                eps_pred,_ = self.diffusion._split_output(model_output["pred"])
                losses = self.criterion(eps_pred,noise)
                loss = losses["loss"]
                if self.use_elt and self.distill is not None:
                    history = model_output["history"]
                    history_pred = {k: self.raw_model.final_layer(v, model_output["c"])for k,v in history.items()}
                    lmax = max(history_pred.keys())
                    teacher = history_pred[lmax].detach()
                    distill_loss = torch.tensor(0.0,device=self.device)
                    for k, student in history_pred.items():
                        if k == lmax:
                            continue
                        distill_loss += self.distill(student,teacher)
                    elt_lambda = self.distill_scheduler(step)
                    losses["elt"]=distill_loss
                    loss = loss + elt_lambda * distill_loss
                if  self.repa is not None:
                    dit_features = self.raw_model.forward_features(x_t,t,labels)
                    if isinstance(dit_features,tuple):
                        dit_features = dit_features[0]
                    with torch.no_grad():
                        target_features = self.repa_encoder(images)
                    repa_loss = self.repa(dit_features,target_features)
                    losses["repa"] = repa_loss
                    loss = loss+self.cfg["repa"]["lambda"]*repa_loss

            self.optimizer.zero_grad(set_to_none=True)
            self.accelerator.backward(loss)
            self.accelerator.clip_grad_norm_(self.model.parameters(),self.cfg["train"]["grad_clip_norm"])
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
        if self.ema is not None:
            self.ema.update(self.raw_model,step=step)
        return {
            "losses": losses,
            "batch_size": images.size(0),
            "history": model_output["history"]
        }

    def validate(self, step):
        if step % self.cfg["eval"]["every"] != 0:
            return
        if not self.accelerator.is_main_process:
            return
        
        results = self.run_evaluation(step)
        current = results["fid"]["metrics"]["fid"]
        if current < self.best_metric:
            self.best_metric = current
            self.best_step = step
            self.best_state = {
                "model": copy.deepcopy(self.raw_model.state_dict()),
                "ema": copy.deepcopy(self.ema.shadow) if self.ema is not None else None,
            }
            print(f"step {step}: new best fid={current:.4f} (will save at end of training)")
                    
    def sample(self):
        sampler = build_sampler(cfg=self.cfg,model=self.raw_model,device=self.device,vae=self.vae,diffusion=self.diffusion,scaling_factor=self.scaling_factor)
        outputs = sampler.generate()
        return outputs["diffusion"]["images"]

    def maybe_sample(self, step):
        if not self.cfg["sampling"]["enabled"]:
            return
        if step % self.cfg["sampling"]["every"] != 0:
            return
        if not self.accelerator.is_main_process:
            return
        backup = self._use_ema()
        try:
            self.raw_model.eval()
            images = self.sample()
            images = denormalize(images)
            wandb.log({"samples": [wandb.Image(img)for img in images],"step": step,})
        finally:
            self._restore_ema(backup)
            self.raw_model.train()

    def log(self, step, train_output):
        self.update_running_losses(train_output["losses"],train_output["batch_size"])
        if step % self.cfg["train"]["log_every"] != 0:
            return
        metrics = self.get_running_metrics()
        if self.accelerator.is_main_process:
            current_lr = self.scheduler.get_last_lr()[0] if self.scheduler is not None else self.optimizer.param_groups[0]["lr"]
            wandb.log({**{f"train/{k}": v for k,v in metrics.items()}, "train/lr": current_lr, "step": step})
            if step % 10000 == 0:
                print(f"step {step} | lr: {current_lr:.6e}")
            print(f"step {step} | "+" ".join(f"{k}: {v:.5f}"for k,v in metrics.items()))
        self.reset_running_losses()
        
    def save_checkpoint(self,step):
        if step % self.cfg["train"]["ckpt_every"] !=0:
            return
        if not self.accelerator.is_main_process:
            return
        path = os.path.join(self.checkpoint_dir, f"{self.cfg['model']['name']}_run{self.cfg['run_number']}_{step}.pt")
        save_checkpoint(path=path,model=self.raw_model,optimizer=self.optimizer,scheduler=self.scheduler,ema=self.ema,epoch=step,cfg=self.cfg)

    def save_best_checkpoint(self):
        if not self.accelerator.is_main_process:
            return
        if not hasattr(self, "best_state"):
            return
        path = os.path.join(self.checkpoint_dir, f"{self.cfg['model']['name']}_run{self.cfg['run_number']}_best.pt")
        current_state = copy.deepcopy(self.raw_model.state_dict())
        self.raw_model.load_state_dict(self.best_state["model"])
        save_checkpoint(
            path=path, model=self.raw_model, optimizer=self.optimizer,
            scheduler=self.scheduler, ema=self.ema, epoch=self.best_step, cfg=self.cfg,
        )
        self.raw_model.load_state_dict(current_state)

    def save_final_checkpoint(self):
        if not self.accelerator.is_main_process:
            return
        path = os.path.join(self.checkpoint_dir,f"{self.cfg['model']['name']}_run{self.cfg['run_number']}_final.pt")
        save_checkpoint(path=path,model=self.raw_model,optimizer=self.optimizer,scheduler=self.scheduler,ema=self.ema,epoch=self.total_steps,cfg=self.cfg)

    def train(self):
        self.setup()
        for step in range(self.start_step,self.total_steps):
            batch = next(self.train_iter)
            train_output = self.train_step(batch,step)
            self.log(step,train_output)
            self.validate(step)
            self.maybe_sample(step)
            self.save_checkpoint(step)
        self.save_final_checkpoint()
        self.save_best_checkpoint()

# DIT FACTORY
def build_dit_trainer(cfg):
    if cfg["elt"]["enabled"] and cfg["model"]["name"] != "looped_dit":
        raise ValueError(
            "ELT requires model.name='looped_dit'"
        )
    set_seed(cfg["seed"])
    checkpoint_dir = setup_environment(cfg)
    accelerator = build_accelerator(cfg)
    device = accelerator.device
     = build_dataloader(cfg["data"],split="train")
    test_loader = build_dataloader(cfg["data"],split="test")
    loaders logger = build_logger(cfg, accelerator)
    # data
    train_loader= {"train": train_loader,"test": test_loader,}
    # models
    model = build_model(cfg)
    vae= build_vae_from_checkpoint(cfg["vae"]["checkpoint"],device=device,freeze=True)
    scaling_factor = compute_scaling_factor(vae, cfg, split="train", device=device)
    print(f"scaling_factor (computed fresh): {scaling_factor}")
    diffusion = build_diffusion(cfg)
    # training objects
    criterion = build_loss(cfg)
    optimizer = build_optimizer(model,cfg)
    scheduler = build_scheduler(optimizer,cfg)
    window = cfg["train"]["ema_window_ratio"] * cfg["train"]["total_steps"]
    ema_decay = 1 - 1 / window
    ema = EMA(model, decay=ema_decay)
    # evaluation dependency
    fid_metric = build_fid_metric(cfg,device)
    evaluators = build_evaluators(cfg,model,loaders,device,vae=vae,diffusion=diffusion,fid_metric=fid_metric,scaling_factor=scaling_factor)
    distill = build_distillation(cfg)
    repa, repa_encoder = build_repa(cfg, model)


    return DiTTrainer(cfg,model,vae,diffusion,optimizer,criterion,train_loader,accelerator,device,checkpoint_dir,scheduler=scheduler,ema=ema,logger=logger,evaluators=evaluators,distill=distill,repa = repa,repa_encoder = repa_encoder,scaling_factor=scaling_factor)

TRAINER_BUILDERS = {
    "vae": build_vae_trainer,
    "dit": build_dit_trainer,
}

def build_trainer(cfg):
    name = cfg["model"]["name"]
    if name not in TRAINER_BUILDERS:
        raise ValueError(f"Unknown trainer type '{name}', expected one of {list(TRAINER_BUILDERS)}")
    return TRAINER_BUILDERS[name](cfg)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train an ELT model.")
    parser.add_argument("--config", default="configs/default.yaml", help="Base training configuration")
    parser.add_argument("--model-config", required=True, help="Model configuration file")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--max-steps", type=int, default=None, help="Override maximum training steps")
    parser.add_argument("--resume", default=None, help="Checkpoint path to resume training from")

    args = parser.parse_args()
    cfg = load_config(args.config, args.model_config)

    if args.batch_size is not None:
        cfg["data"]["batch_size"] = args.batch_size
    if args.lr is not None:
        cfg["optimizer"]["lr"] = args.lr
    if args.max_steps is not None:
        cfg["train"]["total_steps"] = args.max_steps
    if args.resume is not None:
        cfg["train"]["resume"] = args.resume

    trainer = build_trainer(cfg)
    trainer.train()