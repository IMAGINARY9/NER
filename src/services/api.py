"""
Module for Entity Extraction API services.
"""

import logging
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional

from src.predict import NERPredictor

logger = logging.getLogger(__name__)

# Define data models
class EntityRequest(BaseModel):
    text: str

class Entity(BaseModel):
    text: str
    type: str
    start: int
    end: int

class EntityResponse(BaseModel):
    entities: List[Entity]
    tokens: List[str]
    tags: List[str]


class NERService:
    """Service for Named Entity Recognition"""
    
    def __init__(self, model_path, word_tokenizer, tag_vocab):
        """Initialize the service with a trained model."""
        self.predictor = NERPredictor(model_path, word_tokenizer, tag_vocab)
        logger.info("NER service initialized")
    
    def extract_entities(self, text):
        """
        Extract entities from text.
        
        Args:
            text: String text to extract entities from
            
        Returns:
            dict: Dictionary with extracted entities and token info
        """
        try:
            tokens, tags, entities = self.predictor.predict(text)
            
            return {
                "entities": entities,
                "tokens": tokens,
                "tags": tags
            }
        except Exception as e:
            logger.error(f"Error extracting entities: {e}")
            raise


def create_app(service):
    """Create a FastAPI application for the NER service."""
    app = FastAPI(
        title="Named Entity Recognition API",
        description="API for extracting entities from text",
        version="1.0.0"
    )
    
    @app.post("/extract", response_model=EntityResponse)
    async def extract_entities(request: EntityRequest):
        try:
            result = service.extract_entities(request.text)
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/health")
    async def health_check():
        return {"status": "healthy"}
    
    return app
