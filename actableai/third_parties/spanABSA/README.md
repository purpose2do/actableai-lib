# Package spanabsa 


1. Get model:  
from actableai.third_parties.spanABSA import download_model  
download_model.download_spanabsa()
2. Use:  
from actableai.sentiment.SpanABSA import SpanABSA  
model = SpanABSA()  
sentences = ["this puck is good"]  
results = model.predict(sentences)