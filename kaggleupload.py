import kagglehub

# For example, to upload a new variation to this model:
# - https://www.kaggle.com/models/google/bert/frameworks/tensorFlow2/variations/answer-equivalence-bem
# 
# You would use the following handle: `google/bert/tensorFlow2/answer-equivalence-bem`
handle = 'mhdmaen/excavating-text/Transformers/swedish-arch-ner'
local_model_dir = './kbtraining/checkpoint-15000/'

kagglehub.model_upload(handle, local_model_dir)

# You can also specify some version notes (optional)
#kagglehub.model_upload(handle, local_model_dir, version_notes='improved accuracy')

# You can also specify a license (optional)
kagglehub.model_upload(handle, local_model_dir, license_name='Apache 2.0')
