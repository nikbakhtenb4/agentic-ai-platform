
# ===============================
# services/llm-service/models/text_generation.py
# ===============================
import torch
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class TextGenerator:
    def __init__(self, model_loader):
        self.model_loader = model_loader
    
    async def generate(
        self,
        prompt: str,
        max_length: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        num_return_sequences: int = 1
    ) -> Dict[str, Any]:
        """Generate text from prompt"""
        
        if not self.model_loader.is_loaded:
            raise RuntimeError("Model not loaded")
        
        try:
            # Tokenize input
            inputs = self.model_loader.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            
            # Move to device
            inputs = {k: v.to(self.model_loader.device) for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = self.model_loader.model.generate(
                    **inputs,
                    max_length=inputs['input_ids'].shape[1] + max_length,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample,
                    num_return_sequences=num_return_sequences,
                    pad_token_id=self.model_loader.tokenizer.eos_token_id,
                    eos_token_id=self.model_loader.tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            # Decode outputs
            generated_texts = []
            total_tokens = 0
            
            for output in outputs:
                # Remove input tokens from output
                generated_tokens = output[inputs['input_ids'].shape[1]:]
                generated_text = self.model_loader.tokenizer.decode(
                    generated_tokens, 
                    skip_special_tokens=True
                )
                generated_texts.append(generated_text.strip())
                total_tokens += len(generated_tokens)
            
            return {
                "generated_text": generated_texts,
                "model_name": self.model_loader.model_name,
                "token_count": total_tokens,
                "gpu_used": self.model_loader.device == "cuda"
            }
            
        except Exception as e:
            logger.error(f"Generation error: {e}")
            raise




# # services/llm-service/models/text_generation.py
# import asyncio
# import logging
# from typing import Dict, List, Optional, Any, AsyncGenerator
# import torch
# from transformers import (
#     AutoTokenizer,
#     AutoModelForCausalLM,
#     GenerationConfig,
#     TextStreamer
# )
# import uuid
# from datetime import datetime
# import json
# import re

# logger = logging.getLogger(__name__)

# class TextGenerator:
#     """کلاس تولید متن با مدل‌های فارسی"""
    
#     def __init__(self, model_loader):
#         self.model_loader = model_loader
#         self.conversations = {}  # ذخیره مکالمات
#         self.generation_config = None
#         self._setup_generation_config()
    
#     def _setup_generation_config(self):
#         """تنظیم پیکربندی تولید متن"""
#         self.generation_config = GenerationConfig(
#             max_new_tokens=512,
#             temperature=0.7,
#             top_p=0.9,
#             top_k=50,
#             do_sample=True,
#             pad_token_id=0,
#             eos_token_id=2,
#             repetition_penalty=1.1,
#             length_penalty=1.0,
#             no_repeat_ngram_size=3
#         )
    
#     async def generate(
#         self,
#         prompt: str,
#         max_tokens: int = 512,
#         temperature: float = 0.7,
#         top_p: float = 0.9,
#         conversation_id: Optional[str] = None,
#         stream: bool = False
#     ) -> Dict[str, Any]:
#         """تولید پاسخ برای prompt ورودی"""
        
#         try:
#             # Get model and tokenizer
#             model_name = "persian-llm"
#             model_info = self.model_loader.get_model(model_name)
            
#             if not model_info:
#                 raise ValueError(f"Model {model_name} not loaded")
            
#             model = model_info["model"]
#             tokenizer = model_info["tokenizer"]
            
#             # Handle conversation context
#             if conversation_id:
#                 full_prompt = self._build_conversation_prompt(prompt, conversation_id)
#             else:
#                 conversation_id = str(uuid.uuid4())
#                 full_prompt = self._format_persian_prompt(prompt)
            
#             # Update generation config
#             gen_config = GenerationConfig(
#                 max_new_tokens=max_tokens,
#                 temperature=temperature,
#                 top_p=top_p,
#                 top_k=50,
#                 do_sample=True,
#                 pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
#                 eos_token_id=tokenizer.eos_token_id,
#                 repetition_penalty=1.1,
#                 length_penalty=1.0,
#                 no_repeat_ngram_size=3
#             )
            
#             if stream:
#                 return await self._generate_stream(
#                     model, tokenizer, full_prompt, gen_config, conversation_id
#                 )
#             else:
#                 return await self._generate_single(
#                     model, tokenizer, full_prompt, gen_config, conversation_id, prompt
#                 )
                
#         except Exception as e:
#             logger.error(f"خطا در تولید متن: {e}")
#             raise
    
#     async def _generate_single(
#         self,
#         model,
#         tokenizer,
#         prompt: str,
#         gen_config: GenerationConfig,
#         conversation_id: str,
#         original_prompt: str
#     ) -> Dict[str, Any]:
#         """تولید پاسخ یکباره"""
        
#         start_time = asyncio.get_event_loop().time()
        
#         # Tokenize input
#         inputs = tokenizer(
#             prompt,
#             return_tensors="pt",
#             truncation=True,
#             max_length=2048,
#             padding=True
#         )
        
#         if torch.cuda.is_available():
#             inputs = {k: v.cuda() for k, v in inputs.items()}
        
#         input_length = inputs["input_ids"].shape[1]
        
#         # Generate
#         with torch.no_grad():
#             outputs = model.generate(
#                 **inputs,
#                 generation_config=gen_config,
#                 return_dict_in_generate=True,
#                 output_scores=True
#             )
        
#         # Decode output
#         generated_ids = outputs.sequences[0][input_length:]
#         response_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
#         # Clean up response
#         response_text = self._clean_response(response_text)
        
#         # Update conversation
#         self._update_conversation(conversation_id, original_prompt, response_text)
        
#         generation_time = asyncio.get_event_loop().time() - start_time
#         tokens_used = len(generated_ids)
        
#         return {
#             "text": response_text,
#             "conversation_id": conversation_id,
#             "tokens_used": tokens_used,
#             "model_name": "persian-llm",
#             "generation_time": generation_time,
#             "metadata": {
#                 "input_tokens": input_length,
#                 "output_tokens": tokens_used,
#                 "total_tokens": input_length + tokens_used,
#                 "temperature": gen_config.temperature,
#                 "top_p": gen_config.top_p
#             }
#         }
    
#     async def _generate_stream(
#         self,
#         model,
#         tokenizer,
#         prompt: str,
#         gen_config: GenerationConfig,
#         conversation_id: str
#     ) -> AsyncGenerator[Dict[str, Any], None]:
#         """تولید پاسخ جریانی"""
        
#         # این قسمت برای streaming response ها
#         # فعلاً ساده پیاده‌سازی شده، می‌تواند پیچیده‌تر شود
        
#         inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        
#         if torch.cuda.is_available():
#             inputs = {k: v.cuda() for k, v in inputs.items()}
        
#         input_length = inputs["input_ids"].shape[1]
        
#         # For now, return single response (can be improved for true streaming)
#         result = await self._generate_single(
#             model, tokenizer, prompt, gen_config, conversation_id, prompt
#         )
        
#         yield {
#             "chunk": result["text"],
#             "conversation_id": conversation_id,
#             "finished": True,
#             "metadata": result["metadata"]
#         }
    
#     def _format_persian_prompt(self, prompt: str) -> str:
#         """فرمت کردن prompt برای مدل فارسی"""
        
#         # Template for Persian instruction-following
#         template = """### دستورالعمل:
# {instruction}

# ### پاسخ:
# """
        
#         return template.format(instruction=prompt.strip())
    
#     def _build_conversation_prompt(self, prompt: str, conversation_id: str) -> str:
#         """ساخت prompt با context مکالمه"""
        
#         conversation = self.conversations.get(conversation_id, [])
        
#         if not conversation:
#             return self._format_persian_prompt(prompt)
        
#         # Build conversation context
#         context_parts = []
        
#         # Add previous turns (limit to last 3 turns to avoid context overflow)
#         recent_turns = conversation[-3:] if len(conversation) > 3 else conversation
        
#         for turn in recent_turns:
#             context_parts.append(f"### سوال: {turn['user']}")
#             context_parts.append(f"### پاسخ: {turn['assistant']}")
        
#         # Add current question
#         context_parts.append(f"### سوال: {prompt}")
#         context_parts.append("### پاسخ:")
        
#         return "\n\n".join(context_parts)
    
#     def _clean_response(self, text: str) -> str:
#         """پاکسازی پاسخ تولید شده"""
        
#         # Remove common artifacts
#         text = text.strip()
        
#         # Remove repeated patterns
#         text = re.sub(r'(.+?)\1{2,}', r'\1', text)
        
#         # Remove incomplete sentences at the end
#         sentences = text.split('.')
#         if len(sentences) > 1 and len(sentences[-1].strip()) < 10:
#             text = '.'.join(sentences[:-1]) + '.'
        
#         # Remove common generation artifacts
#         artifacts = [
#             "### دستورالعمل:",
#             "### پاسخ:",
#             "### سوال:",
#             "<pad>",
#             "<s>",
#             "</s>",
#             "<unk>"
#         ]
        
#         for artifact in artifacts:
#             text = text.replace(artifact, "").strip()
        
#         return text
    
#     def _update_conversation(self, conversation_id: str, user_message: str, assistant_message: str):
#         """به‌روزرسانی تاریخچه مکالمه"""
        
#         if conversation_id not in self.conversations:
#             self.conversations[conversation_id] = []
        
#         self.conversations[conversation_id].append({
#             "user": user_message,
#             "assistant": assistant_message,
#             "timestamp": datetime.now().isoformat()
#         })
        
#         # Keep only last 10 turns to manage memory
#         if len(self.conversations[conversation_id]) > 10:
#             self.conversations[conversation_id] = self.conversations[conversation_id][-10:]
    
#     def get_conversation_history(self, conversation_id: str) -> List[Dict[str, Any]]:
#         """دریافت تاریخچه مکالمه"""
#         return self.conversations.get(conversation_id, [])
    
#     def clear_conversation(self, conversation_id: str):
#         """پاک کردن تاریخچه مکالمه"""
#         if conversation_id in self.conversations:
#             del self.conversations[conversation_id]
    
#     def get_active_conversations(self) -> List[str]:
#         """دریافت لیست مکالمات فعال"""
#         return list(self.conversations.keys())
    
#     async def batch_generate(
#         self,
#         prompts: List[str],
#         max_tokens: int = 512,
#         temperature: float = 0.7,
#         top_p: float = 0.9
#     ) -> List[Dict[str, Any]]:
#         """تولید پاسخ برای چندین prompt همزمان"""
        
#         tasks = []
#         for prompt in prompts:
#             task = self.generate(
#                 prompt=prompt,
#                 max_tokens=max_tokens,
#                 temperature=temperature,
#                 top_p=top_p
#             )
#             tasks.append(task)
        
#         results = await asyncio.gather(*tasks, return_exceptions=True)
        
#         # Handle exceptions
#         processed_results = []
#         for i, result in enumerate(results):
#             if isinstance(result, Exception):
#                 processed_results.append({
#                     "error": str(result),
#                     "prompt_index": i,
#                     "text": "خطا در تولید پاسخ"
#                 })
#             else:
#                 processed_results.append(result)
        
#         return processed_results