## Adapted by: Kyler Smith
### This file establishes the API for AudioLDM2 training and finetuning.
### It is a class that can be instantiated and used to manipulate config variables before training.

# imports

import sys

sys.path.append("src")
import shutil
import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"

import argparse
import yaml
import torch
import zipfile

from tqdm import tqdm
from pytorch_lightning.strategies.ddp import DDPStrategy
from audioldm_train.utilities.data.dataset import AudioDataset
from audioldm_train.utilities.tools import build_dataset_json_from_list

from torch.utils.data import DataLoader
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from audioldm_train.utilities.tools import (
    get_restore_step,
    copy_test_subset_data,
)
from audioldm_train.utilities.model_util import instantiate_from_config
import logging

logging.basicConfig(level=logging.WARNING)

## Data Processing imports
import audioldm_train.utilities.processFromZip as processFromZip


## Create an API class that can hold an instance of all the settings we need
class AudioLDM2APIObject:
    def __init__(
        self,
        configs=None,
        config_yaml_path="audioldm_train/config/2025_03_27_api_default_finetune/default_finetune.yaml",
        perform_validation=False,
    ):

        # assert torch.cuda.is_available(), "CUDA is not available. API failed to initialize."

        print("Initializing AudioLDM2 API...")

        self.perform_validation = perform_validation

        # Parse yaml path into experiment names and config path
        self.exp_name = os.path.basename(config_yaml_path.split(".")[0])
        self.exp_group_name = os.path.basename(os.path.dirname(config_yaml_path))

        self.config_yaml_path = os.path.join(config_yaml_path)

        if configs is not None:
            self.configs = configs
        else:
            self.configs = yaml.load(
                open(self.config_yaml_path, "r"), Loader=yaml.FullLoader
            )

        if perform_validation:
            self.__performValidation()

        ## Variables to be shared between functions
        self.resume_from_checkpoint = None
        self.test_data_subset_folder = None

        self.dataset = None
        self.dataloader = None
        self.val_dataset = None
        self.val_dataloader = None
        self.checkpoint_callback = None
        self.latent_diffusion = None
        self.wandb_logger = None
        self.trainer = None

    def setReloadFromCheckpoint(self, flag):
        self.configs["reload_from_ckpt"] = flag

    def __performValidation(self):
        self.configs["model"]["params"]["cond_stage_config"][
            "crossattn_audiomae_generated"
        ]["params"]["use_gt_mae_output"] = False
        self.configs["step"]["limit_val_batches"] = None

    def handleDataUpload(self, zipPath):
        # Note that this could be written to set metadata_root depending on where processFromZip is configured to extract things
        ## but for now processFromZip universally makes it match the audioset formatting, so this is nonessential
        return processFromZip.process(zipPath)

    # Function to return the path of a checkpoint which can be downloaded by the user
    ## Helper function for a client-side implementation of a file download
    def prepareCheckpointDownload(self):
        # Search through the checkpoints directory for the most recent checkpoint
        checkpointDir = os.path.join(
            self.configs["log_directory"],
            self.exp_group_name,
            self.exp_name,
            "checkpoints",
        )
        files = os.listdir(checkpointDir)
        paths = []
        for basename in files:
            if basename.endswith(".ckpt"):
                paths.append(os.path.join(checkpointDir, basename))
        checkpointPath = max(paths, key=os.path.getctime)
        # Compress this checkpoint
        compressedPath = os.path.splitext(checkpointPath)[0] + ".zip"
        zipfile.ZipFile(
            compressedPath, compression=zipfile.ZIP_DEFLATED, mode="w"
        ).write(checkpointPath, arcname=os.path.basename(checkpointPath))
        # Return file path of compressed checkpoint
        return compressedPath

    def prepareAllValidationsDownload(self):
        logsDir = os.path.join(
            self.configs["log_directory"], self.exp_group_name, self.exp_name
        )
        subfolders = [
            f.path for f in os.scandir(logsDir) if (f.is_dir() and ("val" in f.name))
        ]

        compressedPath = os.path.join(logsDir, "allValidations.zip")

        with zipfile.ZipFile(
            compressedPath, compression=zipfile.ZIP_DEFLATED, mode="w"
        ) as archive:
            for folder_path in subfolders:
                for root, dirs, files in os.walk(folder_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        archive.write(file_path, os.path.relpath(file_path, logsDir))

    def __readInferencePromptsFile(self, promptsJsonPath):
        # Read in file
        promptsList = []
        with open(promptsJsonPath, "r") as f:
            for each in f.readlines():
                each = each.strip("\n")
                promptsList.append(each)
        # Process allowed filename delimiters
        data = []
        for each in promptsList:
            if "|" in each:
                wav, caption = each.split("|")
            else:
                caption = each
                wav = ""
            data.append(
                {
                    "wav": wav,
                    "caption": caption,
                }
            )
        return {"data": data}

    def __infer(self, dataset_json):
        if "dataloader_add_ons" in self.configs["data"].keys():
            dataloader_add_ons = self.configs["data"]["dataloader_add_ons"]
        else:
            dataloader_add_ons = []

        val_dataset = AudioDataset(
            self.configs,
            split="test",
            add_ons=dataloader_add_ons,
            dataset_json=dataset_json,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
        )

        try:
            config_reload_from_ckpt = self.configs["reload_from_ckpt"]
        except:
            config_reload_from_ckpt = "data/checkpoints/audioldm-m-full.ckpt"

        self.setReloadFromCheckpoint(True)

        checkpoint_path = os.path.join(
            self.configs["log_directory"],
            self.exp_group_name,
            self.exp_name,
            "checkpoints",
        )

        wandb_path = os.path.join(
            self.configs["log_directory"], self.exp_group_name, self.exp_name
        )

        os.makedirs(checkpoint_path, exist_ok=True)
        shutil.copy(self.config_yaml_path, wandb_path)

        if len(os.listdir(checkpoint_path)) > 0:
            print("Load checkpoint from path: %s" % checkpoint_path)
            restore_step, n_step = get_restore_step(checkpoint_path)
            resume_from_checkpoint = os.path.join(checkpoint_path, restore_step)
            print("Resume from checkpoint", resume_from_checkpoint)
        elif config_reload_from_ckpt is not None:
            resume_from_checkpoint = config_reload_from_ckpt
            print(
                "Reload ckpt specified in the config file %s" % resume_from_checkpoint
            )
        else:
            print("Train from scratch")
            resume_from_checkpoint = None

        self.latent_diffusion = instantiate_from_config(self.configs["model"])
        self.latent_diffusion.set_log_dir(
            self.configs["log_directory"], self.exp_group_name, self.exp_name
        )

        checkpoint = torch.load(resume_from_checkpoint)
        self.latent_diffusion.load_state_dict(checkpoint["state_dict"])

        self.latent_diffusion.eval()
        self.latent_diffusion = self.latent_diffusion.cuda()

        return self.latent_diffusion.generate_sample(
            val_loader,
            unconditional_guidance_scale=self.configs["model"]["params"][
                "evaluation_params"
            ]["unconditional_guidance_scale"],
            ddim_steps=self.configs["model"]["params"]["evaluation_params"][
                "ddim_sampling_steps"
            ],
            n_gen=self.configs["model"]["params"]["evaluation_params"][
                "n_candidates_per_samples"
            ],
        )

    def inferSingle(self, prompt):
        # Create json object
        dataset_json = {
            "data": {
                "wav": "",
                "caption": prompt,
            }
        }
        return self.__infer(dataset_json)

    def inferFromFile(self, promptsJsonPath):
        dataset_json = self.__readInferencePromptsFile(promptsJsonPath)
        return self.__infer(dataset_json)

    def __initializeSystemSettings(self):
        ## use self.configs to initialize system settings
        if "seed" in self.configs.keys():
            seed_everything(self.configs["seed"])
        else:
            print("SEED EVERYTHING TO 0")
            seed_everything(0)

        if "precision" in self.configs.keys():
            torch.set_float32_matmul_precision(
                self.configs["precision"]
            )  # precision can be highest, high, medium

    def __initializeDatasetSplits(self):
        # catch missing dataloader_add_ons
        if "dataloader_add_ons" in self.configs["data"].keys():
            dataloader_add_ons = self.configs["data"]["dataloader_add_ons"]
        else:
            dataloader_add_ons = []

        ## "train" split
        self.dataset = AudioDataset(
            self.configs, split="train", add_ons=dataloader_add_ons
        )

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.configs["model"]["params"][
                "batchsize"
            ],  # Change this for potential changes in training quality -- smaller batches = more stable
            num_workers=16,  # change this for potential changes in speed -- quality shouldn't change
            pin_memory=True,
            shuffle=True,
        )

        print(
            "The length of the dataset is %s, the length of the dataloader is %s, the batchsize is %s"
            % (
                len(self.dataset),
                len(self.dataloader),
                self.configs["model"]["params"]["batchsize"],
            )
        )

        self.val_dataset = AudioDataset(
            self.configs, split="test", add_ons=dataloader_add_ons
        )

        self.val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=8,
        )

    def __copyTestData(self):
        self.test_data_subset_folder = os.path.join(
            os.path.dirname(self.configs["log_directory"]),
            "testset_data",
            self.val_dataset.dataset_name,
        )
        os.makedirs(self.test_data_subset_folder, exist_ok=True)
        copy_test_subset_data(self.val_dataset.data, self.test_data_subset_folder)

    def __initTrainer(self):
        ## init local variables
        try:
            config_reload_from_ckpt = self.configs["reload_from_ckpt"]
        except:
            config_reload_from_ckpt = None

        try:
            limit_val_batches = self.configs["step"]["limit_val_batches"]
        except:
            limit_val_batches = None

        validation_every_n_epochs = self.configs["step"]["validation_every_n_epochs"]
        save_checkpoint_every_n_steps = self.configs["step"][
            "save_checkpoint_every_n_steps"
        ]
        max_steps = self.configs["step"]["max_steps"]
        save_top_k = self.configs["step"]["save_top_k"]

        checkpoint_path = os.path.join(
            self.configs["log_directory"],
            self.exp_group_name,
            self.exp_name,
            "checkpoints",
        )

        wandb_path = os.path.join(
            self.configs["log_directory"], self.exp_group_name, self.exp_name
        )

        self.checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_path,
            monitor="global_step",
            mode="max",
            filename="checkpoint-fad-{val/frechet_inception_distance:.2f}-global_step={global_step:.0f}",
            every_n_train_steps=save_checkpoint_every_n_steps,
            save_top_k=save_top_k,
            auto_insert_metric_name=False,
            save_last=False,
        )

        os.makedirs(checkpoint_path, exist_ok=True)
        shutil.copy(self.config_yaml_path, wandb_path)

        # set self.resume_from_checkpoint
        is_external_checkpoints = False
        if len(os.listdir(checkpoint_path)) > 0:
            print("Load checkpoints from path: %s" % checkpoint_path)
            restore_step, n_step = get_restore_step(checkpoint_path)
            resume_from_checkpoint = os.path.join(checkpoint_path, restore_step)
            print("Resume from checkpoint", resume_from_checkpoint)
        elif config_reload_from_ckpt is not None:
            resume_from_checkpoint = config_reload_from_ckpt
            is_external_checkpoints = True
            print(
                "Reload ckpt specified in the config file %s" % resume_from_checkpoint
            )
        else:
            print("Train from scratch")
            resume_from_checkpoint = None

        devices = torch.cuda.device_count()

        self.latent_diffusion = instantiate_from_config(self.configs["model"])
        self.latent_diffusion.set_log_dir(
            self.configs["log_directory"], self.exp_group_name, self.exp_name
        )

        self.wandb_logger = WandbLogger(
            save_dir=wandb_path,
            project=self.configs["project"],
            config=self.configs,
            name="%s/%s" % (self.exp_group_name, self.exp_name),
        )

        self.latent_diffusion.test_data_subset_path = self.test_data_subset_folder

        print("==> Save checkpoint every %s steps" % save_checkpoint_every_n_steps)
        print("==> Perform validation every %s epochs" % validation_every_n_epochs)

        self.trainer = Trainer(
            accelerator="gpu",
            devices=devices,
            logger=self.wandb_logger,
            max_steps=max_steps,
            num_sanity_val_steps=1,
            limit_val_batches=limit_val_batches,
            check_val_every_n_epoch=validation_every_n_epochs,
            strategy=DDPStrategy(find_unused_parameters=True),
            callbacks=[self.checkpoint_callback],
        )

        return is_external_checkpoints

    ## internal function for running the torch training process
    def __train(self, is_external_checkpoints=False):
        if is_external_checkpoints:
            if self.resume_from_checkpoint is not None:
                ckpt = torch.load(self.resume_from_checkpoint)["state_dict"]

                key_not_in_model_state_dict = []
                size_mismatch_keys = []
                state_dict = self.latent_diffusion.state_dict()
                print("Filtering key for reloading:", self.resume_from_checkpoint)
                print(
                    "State dict key size:",
                    len(list(state_dict.keys())),
                    len(list(ckpt.keys())),
                )
                for key in tqdm(list(ckpt.keys())):
                    if key not in state_dict.keys():
                        key_not_in_model_state_dict.append(key)
                        del ckpt[key]
                        continue
                    if state_dict[key].size() != ckpt[key].size():
                        del ckpt[key]
                        size_mismatch_keys.append(key)

                if len(key_not_in_model_state_dict) != 0:
                    print(
                        "==> Warning: The following key in the checkpoint is not presented in the model:",
                        key_not_in_model_state_dict,
                    )
                if len(size_mismatch_keys) != 0:
                    print(
                        "==> Warning: These keys have different size between checkpoint and current model: ",
                        size_mismatch_keys,
                    )

                self.latent_diffusion.load_state_dict(ckpt, strict=False)

            self.trainer.fit(
                self.latent_diffusion, self.dataloader, self.val_dataloader
            )
        else:
            self.trainer.fit(
                self.latent_diffusion,
                self.dataloader,
                self.val_dataloader,
                ckpt_path=self.resume_from_checkpoint,
            )

    # The full training function, called to both train from scratch and to finetune
    # Modularized for better clarity
    ## Reads self.configs["reload_from_ckpt"] to determine if it should reload from a checkpoint (thus finetuning)
    def __beginTrain(self):
        self.__initializeSystemSettings()
        self.__initializeDatasetSplits()
        self.__copyTestData()
        is_external_checkpoints = self.__initTrainer()
        self.__train(is_external_checkpoints)

    def finetune(self):
        self.setReloadFromCheckpoint(True)
        return self.__beginTrain()

    def trainFromScratch(self):
        self.setReloadFromCheckpoint(False)
        return self.__beginTrain()

    ## Parameter setting functions

    def set_save_checkpoint_every_n_steps(self, val):
        self.configs["step"]["save_checkpoint_every_n_steps"] = 5000

    # takes list of keys to drop-down and set targetParam
    def set_parameter(self, targetParam, val):

        # Recursive loop to drill down into desired parameter
        def set_nested_dict_value(d, keys, value):
            for key in keys[:-1]:
                d = d.setdefault(key, {})
            d[keys[-1]] = value

        # Catch to make sure there is an array targetParam
        if len(targetParam) <= 0:
            return False

        # Run the recursion
        set_nested_dict_value(self.configs, targetParam, val)

        return True

    def debugFunc(self):
        print("debug message")
