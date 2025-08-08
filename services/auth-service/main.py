# # services/auth-service/main.py
# #!/usr/bin/env python3
# """
# Authentication Service - Simple JWT Authentication
# سرویس احراز هویت ساده
# """

# import os
# import logging
# from datetime import datetime, timedelta
# from typing import Optional, Dict, Any
# from fastapi import FastAPI, HTTPException, Depends, status
# from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# import jwt
# import hashlib
# import uvicorn

# # Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# )
# logger = logging.getLogger(__name__)

# # FastAPI app
# app = FastAPI(
#     title="Authentication Service",
#     description="Simple JWT-based authentication service",
#     version="1.0.0"
# )

# # CORS middleware
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Security
# security = HTTPBearer()

# # Configuration
# SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
# ALGORITHM = "HS256"
# ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))

# # Simple in-memory user store (replace with database in production)
# USERS_DB = {
#     "admin": {
#         "username": "admin",
#         "password": hashlib.sha256("admin123".encode()).hexdigest(),
#         "role": "admin",
#         "active": True
#     },
#     "user": {
#         "username": "user",
#         "password": hashlib.sha256("user123".encode()).hexdigest(),
#         "role": "user",
#         "active": True
#     }
# }

# # Pydantic models
# class LoginRequest(BaseModel):
#     username: str
#     password: str

# class TokenResponse(BaseModel):
#     access_token: str
#     token_type: str
#     expires_in: int

# class UserResponse(BaseModel):
#     username: str
#     role: str
#     active: bool

# # Utility functions
# def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
#     """Create JWT access token"""
#     to_encode = data.copy()
#     if expires_delta:
#         expire = datetime.utcnow() + expires_delta
#     else:
#         expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

#     to_encode.update({"exp": expire})
#     encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
#     return encoded_jwt

# def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
#     """Verify JWT token"""
#     try:
#         payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
#         username: str = payload.get("sub")

#         if username is None:
#             raise HTTPException(
#                 status_code=status.HTTP_401_UNAUTHORIZED,
#                 detail="Could not validate credentials",
#                 headers={"WWW-Authenticate": "Bearer"},
#             )

#         # Check if user still exists and is active
#         user = USERS_DB.get(username)
#         if not user or not user["active"]:
#             raise HTTPException(
#                 status_code=status.HTTP_401_UNAUTHORIZED,
#                 detail="User not found or inactive",
#                 headers={"WWW-Authenticate": "Bearer"},
#             )

#         return {"username": username, "role": user["role"]}

#     except jwt.ExpiredSignatureError:
#         raise HTTPException(
#             status_code=status.HTTP_401_UNAUTHORIZED,
#             detail="Token has expired",
#             headers={"WWW-Authenticate": "Bearer"},
#         )
#     except jwt.JWTError:
#         raise HTTPException(
#             status_code=status.HTTP_401_UNAUTHORIZED,
#             detail="Could not validate credentials",
#             headers={"WWW-Authenticate": "Bearer"},
#         )

# def authenticate_user(username: str, password: str) -> Optional[dict]:
#     """Authenticate user credentials"""
#     user = USERS_DB.get(username)
#     if not user:
#         return None

#     password_hash = hashlib.sha256(password.encode()).hexdigest()
#     if password_hash != user["password"]:
#         return None

#     if not user["active"]:
#         return None

#     return user

# # Routes
# @app.get("/health")
# async def health_check():
#     """Health check endpoint"""
#     return {
#         "status": "healthy",
#         "service": "auth-service",
#         "timestamp": datetime.utcnow().isoformat()
#     }

# @app.post("/login", response_model=TokenResponse)
# async def login(login_data: LoginRequest):
#     """Login endpoint"""
#     user = authenticate_user(login_data.username, login_data.password)

#     if not user:
#         raise HTTPException(
#             status_code=status.HTTP_401_UNAUTHORIZED,
#             detail="Incorrect username or password",
#             headers={"WWW-Authenticate": "Bearer"},
#         )

#     access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
#     access_token = create_access_token(
#         data={"sub": user["username"], "role": user["role"]},
#         expires_delta=access_token_expires
#     )

#     logger.info(f"User {user['username']} logged in successfully")

#     return {
#         "access_token": access_token,
#         "token_type": "bearer",
#         "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60
#     }

# @app.get("/verify", response_model=UserResponse)
# async def verify_token_endpoint(current_user: dict = Depends(verify_token)):
#     """Verify token and return user info"""
#     user = USERS_DB[current_user["username"]]
#     return UserResponse(
#         username=user["username"],
#         role=user["role"],
#         active=user["active"]
#     )

# @app.get("/me", response_model=UserResponse)
# async def get_current_user(current_user: dict = Depends(verify_token)):
#     """Get current user information"""
#     user = USERS_DB[current_user["username"]]
#     return UserResponse(
#         username=user["username"],
#         role=user["role"],
#         active=user["active"]
#     )

# @app.post("/logout")
# async def logout(current_user: dict = Depends(verify_token)):
#     """Logout endpoint (token blacklisting would be implemented here)"""
#     logger.info(f"User {current_user['username']} logged out")
#     return {"message": "Successfully logged out"}

# # Admin endpoints
# @app.get("/users")
# async def list_users(current_user: dict = Depends(verify_token)):
#     """List all users (admin only)"""
#     if current_user["role"] != "admin":
#         raise HTTPException(
#             status_code=status.HTTP_403_FORBIDDEN,
#             detail="Not enough permissions"
#         )

#     users = []
#     for username, user in USERS_DB.items():
#         users.append({
#             "username": user["username"],
#             "role": user["role"],
#             "active": user["active"]
#         })

#     return {"users": users}

# if __name__ == "__main__":
#     # Configuration from environment
#     host = os.getenv("HOST", "0.0.0.0")
#     port = int(os.getenv("PORT", "8004"))
#     log_level = os.getenv("LOG_LEVEL", "info").lower()

#     logger.info(f"🔐 Starting Auth Service on {host}:{port}")
#     logger.info("Default credentials:")
#     logger.info("  Admin: admin/admin123")
#     logger.info("  User: user/user123")

#     uvicorn.run(
#         "main:app",
#         host=host,
#         port=port,
#         log_level=log_level,
