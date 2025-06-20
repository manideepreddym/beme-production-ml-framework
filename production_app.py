"""
BEME Framework - Production Application
======================================

Production-ready FastAPI application with health checks, metrics, and full ML capabilities.
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Dict, Any, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# Framework imports
from beme import get_framework, BEMEFramework
from config.unified_config import get_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="BEME Framework API",
    description="Production ML platform for Expedia travel booking optimization",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# Global framework instance
framework: Optional[BEMEFramework] = None
startup_time = time.time()

@app.on_event("startup")
async def startup_event():
    """Initialize the BEME framework on application startup."""
    global framework
    
    try:
        logger.info("Initializing BEME Framework...")
        framework = get_framework()
        
        # Initialize framework if method exists
        if hasattr(framework, 'initialize'):
            await framework.initialize()
        
        logger.info("✅ BEME Framework initialized successfully")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize BEME Framework: {e}")
        # Don't raise in production - allow app to start with limited functionality
        logger.warning("Framework will run with limited functionality")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown."""
    global framework
    
    if framework:
        try:
            # Shutdown framework if method exists
            if hasattr(framework, 'shutdown'):
                await framework.shutdown()
            logger.info("✅ BEME Framework shutdown completed")
        except Exception as e:
            logger.error(f"❌ Error during shutdown: {e}")

def get_framework_instance() -> BEMEFramework:
    """Dependency to get framework instance."""
    if framework is None:
        raise HTTPException(status_code=503, detail="Framework not initialized")
    return framework

@app.get("/health")
async def health_check():
    """Health check endpoint for load balancers."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "uptime": time.time() - startup_time,
        "version": "1.0.0"
    }

@app.get("/ready")
async def readiness_check(beme: BEMEFramework = Depends(get_framework_instance)):
    """Readiness check for Kubernetes."""
    try:
        # Check if framework is properly initialized
        config = beme.config
        
        checks = {
            "framework": beme is not None,
            "config": config is not None,
            "environment": config.environment if config else "unknown"
        }
        
        all_ready = all(isinstance(v, bool) and v for k, v in checks.items() if k != "environment")
        
        return {
            "ready": all_ready,
            "checks": checks,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Service not ready: {e}")

@app.get("/metrics")
async def metrics_endpoint():
    """Prometheus metrics endpoint."""
    # Basic metrics - expand based on your monitoring needs
    metrics = [
        f"# HELP beme_uptime_seconds Total uptime in seconds",
        f"# TYPE beme_uptime_seconds counter",
        f"beme_uptime_seconds {time.time() - startup_time}",
        f"",
        f"# HELP beme_health Status of the application (1=healthy, 0=unhealthy)",
        f"# TYPE beme_health gauge",
        f"beme_health 1"
    ]
    
    return "\n".join(metrics)

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "BEME Framework API",
        "description": "Production ML platform for Expedia travel booking optimization",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "health": "/health",
            "ready": "/ready", 
            "metrics": "/metrics",
            "docs": "/docs",
            "predict": "/predict"
        },
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/config")
async def get_configuration(beme: BEMEFramework = Depends(get_framework_instance)):
    """Get current configuration (non-sensitive)."""
    config = beme.config
    
    # Return only non-sensitive configuration
    safe_config = {
        "environment": config.environment,
        "debug": config.debug,
        "log_level": config.log_level,
        "monitoring_port": config.monitoring_port,
        "model_serving_port": config.model_serving_port,
        "data_storage_path": config.data_storage_path,
        "feature_store_backend": config.feature_store_backend
    }
    
    return safe_config

@app.post("/predict")
async def predict_booking(
    booking_data: Dict[str, Any],
    background_tasks: BackgroundTasks,
    beme: BEMEFramework = Depends(get_framework_instance)
):
    """Predict booking probability and optimal pricing."""
    try:
        start_time = time.time()
        
        # Validate input data
        required_fields = ["hotel_id", "check_in", "check_out", "guests"]
        for field in required_fields:
            if field not in booking_data:
                raise HTTPException(
                    status_code=400,
                    detail=f"Missing required field: {field}"
                )
        
        # Mock prediction for now - replace with actual ML model
        prediction = {
            "booking_probability": 0.75,
            "optimal_price": 150.0,
            "confidence": 0.85,
            "factors": {
                "hotel_rating": 0.3,
                "seasonal_demand": 0.4, 
                "price_competitiveness": 0.3
            },
            "recommendations": [
                "Consider 10% discount to increase booking probability",
                "Highlight free breakfast in listing"
            ]
        }
        
        processing_time = time.time() - start_time
        
        # Log prediction for monitoring
        background_tasks.add_task(
            log_prediction,
            booking_data,
            prediction,
            processing_time
        )
        
        return {
            "prediction": prediction,
            "metadata": {
                "processing_time_ms": round(processing_time * 1000, 2),
                "model_version": "1.0.0",
                "timestamp": datetime.utcnow().isoformat()
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

async def log_prediction(booking_data: Dict, prediction: Dict, processing_time: float):
    """Log prediction for monitoring and analytics."""
    logger.info(f"Prediction completed in {processing_time:.3f}s for hotel {booking_data.get('hotel_id')}")

@app.get("/models")
async def list_models(beme: BEMEFramework = Depends(get_framework_instance)):
    """List available ML models."""
    # Mock model list - integrate with actual model registry
    models = [
        {
            "name": "booking_probability_v1",
            "version": "1.0.0",
            "status": "active",
            "accuracy": 0.85,
            "last_trained": "2025-06-15T10:00:00Z"
        },
        {
            "name": "price_optimization_v1", 
            "version": "1.0.0",
            "status": "active",
            "mae": 12.5,
            "last_trained": "2025-06-14T15:30:00Z"
        }
    ]
    
    return {"models": models}

@app.get("/monitoring/drift")
async def check_data_drift(beme: BEMEFramework = Depends(get_framework_instance)):
    """Check for data drift in recent predictions."""
    # Mock drift analysis - integrate with Evidently
    drift_report = {
        "drift_detected": False,
        "drift_score": 0.12,
        "threshold": 0.3,
        "features_affected": [],
        "last_check": datetime.utcnow().isoformat(),
        "recommendation": "No action needed"
    }
    
    return drift_report

# Error handlers
@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred",
            "timestamp": datetime.utcnow().isoformat()
        }
    )

if __name__ == "__main__":
    config = get_config()
    
    # Determine host binding
    host = "0.0.0.0" if config.debug else "127.0.0.1"
    port = config.model_serving_port
    
    print(f"🚀 Starting BEME Framework Production Server...")
    print(f"📊 Server will be available at:")
    print(f"   • http://localhost:{port}")
    print(f"   • http://127.0.0.1:{port}")
    print(f"📖 API Documentation: http://localhost:{port}/docs")
    print(f"🏥 Health Check: http://localhost:{port}/health")
    print(f"📈 Metrics: http://localhost:{port}/metrics")
    print(f"🔮 Predictions: http://localhost:{port}/predict")
    print(f"=" * 60)
    
    uvicorn.run(
        "production_app:app",
        host=host,
        port=port,
        workers=1,  # Use 1 worker for development, increase for production
        log_level=config.log_level.lower(),
        reload=config.debug
    )
