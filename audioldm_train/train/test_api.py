from audioldm_train.utilities.audioldm2_api import AudioLDM2APIObject

if __name__ == "__main__":
    apiInstance = AudioLDM2APIObject()
    
    ## Assumes this script is being run from the root folder of the repo
    # apiInstance.handleDataUpload("./webapp/cache/HandPicked-Animal-Subset.zip")
    
    apiInstance.set_parameter(["step", "save_checkpoint_every_n_steps"], 1000)
    
    apiInstance.finetune()