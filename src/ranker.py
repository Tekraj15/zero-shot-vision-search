import torch
from PIL import Image
from transformers import BlipProcessor, BlipForImageTextRetrieval
import torch.nn.functional as F

class Ranker:
    def __init__(self, model_name="Salesforce/blip-itm-base-coco"):
        """
        Initialize the Ranker with a BLIP ITM model.
        
        Args:
            model_name (str): The name of the BLIP model.
        """
        self.device = "cpu"
        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
            
        print(f"Loading Ranker model {model_name} on {self.device}...")
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForImageTextRetrieval.from_pretrained(model_name).to(self.device)
        self.model.eval()
        print("Ranker model loaded.")

    def rank(self, query, candidates, top_k=12):
        """
        Re-rank the candidates using BLIP Image-Text Matching.
        
        Args:
            query (str): The search query.
            candidates (list): List of dictionaries. Each dict must have 'image_path' key.
                               It can also have other keys like 'id', 'metadata', etc.
            top_k (int): Number of top results to return after re-ranking.
            
        Returns:
            list: Re-ranked list of candidates.
        """
        if not candidates:
            return []
            
        scores = []
        valid_candidates = []
        
        # Process candidates
        # BLIP can process in batches, but for simplicity and to handle potentially large images,
        # we'll process one by one or in small batches. Given top_k=50, one by one is acceptable but slow.
        # Let's try batching if possible, or just simple loop for robustness first.
        
        print(f"Re-ranking {len(candidates)} candidates...")
        
        for cand in candidates:
            try:
                image_path = cand.get('image_path')
                if not image_path:
                    continue
                    
                image = Image.open(image_path).convert('RGB')
                
                # Prepare inputs
                inputs = self.processor(images=image, text=query, return_tensors="pt").to(self.device)
                
                # Calculate ITM score
                # BLIP ITM head returns logits for [negative, positive] match
                with torch.no_grad():
                    itm_out = self.model(**inputs)
                    itm_scores = F.softmax(itm_out.itm_score, dim=1)
                    score = itm_scores[0][1].item() # Probability of "match"
                
                cand['score'] = score
                valid_candidates.append(cand)
                
            except Exception as e:
                print(f"Error ranking candidate {cand.get('id')}: {e}")
                continue
            
        # Sort by score descending
        ranked_candidates = sorted(valid_candidates, key=lambda x: x['score'], reverse=True)
        
        return ranked_candidates[:top_k]
