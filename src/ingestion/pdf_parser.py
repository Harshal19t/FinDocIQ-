import os
import json
import re
from typing import List, Dict, Any
from datasets import load_dataset
import pdfplumber

class FinancialDataIngestor:
    """Handles parsing financial PDFs and Hugging Face financial datasets."""
    
    def __init__(self, raw_data_dir: str = "data/raw", processed_dir: str = "data/processed"):
        self.raw_data_dir = raw_data_dir
        self.processed_dir = processed_dir
        os.makedirs(self.raw_data_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)

    def clean_financial_text(self, text: str) -> str:
        """Cleans headers, footers, and redundant whitespaces from financial reports."""
        # Remove multiple newlines and tab spaces
        text = re.sub(r'\n+', '\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        # Remove common boilerplate SEC page numbers and disclaimers
        text = re.sub(r'Page \d+ of \d+', '', text, flags=re.IGNORECASE)
        return text.strip()

    def parse_pdf_report(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extracts narrative text and tables page-by-page from a 10-K / earnings PDF."""
        extracted_pages = []
        
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                cleaned_text = self.clean_financial_text(text)
                
                # Extract tables if present
                tables = page.extract_tables()
                
                extracted_pages.append({
                    "page_number": page_num,
                    "text": cleaned_text,
                    "table_count": len(tables),
                    "raw_tables": tables
                })
                
        print(f"Successfully parsed {len(extracted_pages)} pages from {pdf_path}")
        return extracted_pages

    def load_hf_financial_dataset(self, num_samples: int = 500) -> str:
        """Downloads financial QA data from Hugging Face for model fine-tuning."""
        print("Downloading financial dataset from Hugging Face...")
        
        # Modern, script-less Parquet dataset repository
        dataset = load_dataset("FinanceMTEB/financial_phrasebank", split="train")
        
        processed_data = []
        for idx, item in enumerate(dataset):
            if idx >= num_samples:
                break
                
            # Extract sentence and label safely
            sentence = item.get("sentence") or item.get("text", "")
            label = item.get("label", "neutral")
            
            formatted_item = {
                "id": f"fin_qa_{idx}",
                "context": sentence,
                "label": label,
                "instruction": "Analyze the financial statement and determine the market sentiment.",
                "response": f"The financial sentiment for this statement is categorized as: {label}"
            }
            processed_data.append(formatted_item)

        os.makedirs(self.processed_dir, exist_ok=True)
        output_path = os.path.join(self.processed_dir, "financial_qa_train.jsonl")
        
        with open(output_path, "w", encoding="utf-8") as f:
            for entry in processed_data:
                f.write(json.dumps(entry) + "\n")
                
        print(f"Saved {len(processed_data)} preprocessed samples to {output_path}")
        return output_path