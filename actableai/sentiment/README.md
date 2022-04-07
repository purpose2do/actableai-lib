# Package spanabsa 


1. Get model:  
Pretrained model available on s3://actable-ai-machine-learning  
Download 3 model and unzip file bert-base-uncased.zip   
2. Use:  
from actableai.sentiment.SpanABSA import AASentimentExtractor    
model = AASentimentExtractor(  
			device = DEVICE, #cpu or gpu  
			BERT_DIR = BERT_DIR,  
			EXTRACT_MODEL_DIR = EXTRACT_MODEL_DIR,  
			CLASSIFICATION_MODEL_DIR = CLASSIFICATION_MODEL_DIR   
sentences = ["this puck is good"]    
results = model.predict(sentences)    


# API ray serve on superset

1. Call ray serve API:   
""" Predict sentiment paragraph. Each row of df equivalent with each paragraph. """  
curl --request GET {SERVICE_IP}:8000/span_absa_predict \
--data '{"sentences":["this puck is good.this bug is bad"], "row_id":1}'



# Reference
1. Codebase: https://github.com/huminghao16/SpanABSA  
2. Actable codebase: https://github.com/Actable-AI/actableai-ml/tree/feat/span-absa/SpanABSA  
