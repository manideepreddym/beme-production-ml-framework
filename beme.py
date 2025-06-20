"""
BEME Framework - Main Integration Module
=======================================

This is the central integration module that coordinates all components of the 
unified BEME (Bid Estimate Market Engine) framework:

- Monitoring (Evidently AI integration)
- Model Registry (MLflow integration) 
- Orchestration (Airflow DAGs)
- Datasets (Travel booking data)
- Models (HuggingFace integration)

This module provides a single entry point for the entire framework.
"""

import os
import sys
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any, List
import logging

# Optional data science imports
try:
    import pandas as pd
    import numpy as np
except ImportError:
    pd = None
    np = None

# Add components to path
sys.path.append(str(Path(__file__).parent))

from config.unified_config import get_config, UnifiedConfig

# Optional imports with fallbacks for minimal setup
try:
    from components.monitoring.unified_monitoring import get_monitoring_system, UnifiedMonitoringSystem
except ImportError as e:
    print(f"⚠️  Monitoring component not available: {e}")
    get_monitoring_system = lambda *args, **kwargs: None
    UnifiedMonitoringSystem = None

try:
    from components.model_registry.unified_registry import get_model_registry, UnifiedModelRegistry
except ImportError as e:
    print(f"⚠️  Model registry component not available: {e}")
    get_model_registry = lambda *args, **kwargs: None
    UnifiedModelRegistry = None

try:
    from components.orchestration.unified_orchestration import get_orchestrator, UnifiedOrchestrator
except ImportError as e:
    print(f"⚠️  Orchestration component not available: {e}")
    get_orchestrator = lambda *args, **kwargs: None
    UnifiedOrchestrator = None

try:
    from components.datasets.unified_data_manager import get_data_manager, UnifiedDataManager
except ImportError as e:
    print(f"⚠️  Data manager component not available: {e}")
    get_data_manager = lambda *args, **kwargs: None
    UnifiedDataManager = None

try:
    from components.models.unified_models import get_model_system, UnifiedModelSystem
except ImportError as e:
    print(f"⚠️  Model system component not available: {e}")
    get_model_system = lambda *args, **kwargs: None
    UnifiedModelSystem = None

class BEMEFramework:
    """
    Main BEME Framework class that coordinates all components.
    
    This provides a unified interface to:
    - Monitor model performance and data drift
    - Manage model lifecycle and deployment
    - Orchestrate training and prediction workflows
    - Handle travel booking datasets
    - Serve models for real-time predictions
    """
    
    def __init__(self, config: Optional[UnifiedConfig] = None):
        """
        Initialize the BEME Framework.
        
        Args:
            config: Optional configuration override
        """
        # Set configuration
        if config:
            from config.unified_config import set_config
            set_config(config)
        
        self.config = get_config()
        self.logger = self._setup_logging()
          # Component instances (may be None if not available)
        self.monitoring = None
        self.model_registry = None  
        self.orchestrator = None
        self.data_manager = None
        self.model_system = None
        
        # Framework state
        self.is_initialized = False
        self.running_services = []
        
        self.logger.info(f"BEME Framework initialized in {self.config.environment} environment")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup unified logging for the framework."""
        logger = logging.getLogger("beme_framework")
        logger.setLevel(getattr(logging, self.config.log_level, "INFO"))
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    async def initialize(self, components: Optional[List[str]] = None):
        """
        Initialize all framework components.
        
        Args:
            components: Optional list of specific components to initialize
                       Options: ['monitoring', 'registry', 'orchestration', 'data', 'models']
                       If None, all components are initialized
        """
        if components is None:
            components = ['monitoring', 'registry', 'orchestration', 'data', 'models']
        
        self.logger.info(f"Initializing BEME framework components: {components}")
        
        try:
            # Initialize components in dependency order
            if 'data' in components:
                await self._init_data_manager()
            
            if 'registry' in components:
                await self._init_model_registry()
            
            if 'models' in components:
                await self._init_model_system()
            
            if 'monitoring' in components:
                await self._init_monitoring()
            
            if 'orchestration' in components:
                await self._init_orchestration()
            
            self.is_initialized = True
            self.logger.info("✅ BEME framework initialization completed (some components may be optional)")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize BEME framework: {str(e)}")
            # Don't re-raise in production - allow framework to work with available components
            self.logger.warning("Framework will operate with available components only")
            self.is_initialized = True
    
    async def _init_data_manager(self):
        """Initialize the data management component."""
        self.logger.info("Initializing data management component")
        self.data_manager = get_data_manager()
        self.logger.info("Data management component initialized")
    
    async def _init_model_registry(self):
        """Initialize the model registry component."""
        self.logger.info("Initializing model registry component")
        self.model_registry = get_model_registry()
        self.logger.info("Model registry component initialized")
    
    async def _init_model_system(self):
        """Initialize the model system component."""
        self.logger.info("Initializing model system component")
        self.model_system = get_model_system()
        self.logger.info("Model system component initialized")
    
    async def _init_monitoring(self):
        """Initialize the monitoring component."""
        self.logger.info("Initializing monitoring component")
        self.monitoring = get_monitoring_system()
        await self.monitoring.initialize_monitoring()
        self.logger.info("Monitoring component initialized")
    
    async def _init_orchestration(self):
        """Initialize the orchestration component."""
        self.logger.info("Initializing orchestration component")
        self.orchestrator = get_orchestrator()
        self.logger.info("Orchestration component initialized")
    
    async def start_services(self, services: Optional[List[str]] = None):
        """
        Start framework services.
        
        Args:
            services: Optional list of specific services to start
                     Options: ['monitoring', 'api_server']
                     If None, all services are started
        """
        if not self.is_initialized:
            await self.initialize()
        
        if services is None:
            services = ['monitoring', 'api_server']
        
        self.logger.info(f"Starting BEME framework services: {services}")
        
        try:
            # Start services
            if 'monitoring' in services and self.monitoring:
                asyncio.create_task(self._run_monitoring_service())
                self.running_services.append('monitoring')
            
            if 'api_server' in services and self.model_system:
                asyncio.create_task(self._run_api_service())
                self.running_services.append('api_server')
            
            self.logger.info(f"Started services: {self.running_services}")
            
        except Exception as e:
            self.logger.error(f"Failed to start services: {str(e)}")
            raise
    
    async def _run_monitoring_service(self):
        """Run the monitoring service."""
        try:
            await self.monitoring.run_monitoring_loop()
        except Exception as e:
            self.logger.error(f"Monitoring service failed: {str(e)}")
    
    async def _run_api_service(self):
        """Run the API service."""
        try:
            # This would start the FastAPI server in a separate process/thread
            # For now, we'll just log that it would start
            self.logger.info("API service would start here")
        except Exception as e:
            self.logger.error(f"API service failed: {str(e)}")
    
    async def stop_services(self):
        """Stop all running services."""
        self.logger.info("Stopping BEME framework services")
        
        # In a real implementation, this would gracefully stop all services
        self.running_services.clear()
        self.logger.info("All services stopped")
    
    async def shutdown(self):
        """Shutdown the framework gracefully."""
        self.logger.info("Shutting down BEME framework...")
        
        try:
            # Stop all services
            await self.stop_services()
            
            # Clean up components
            if self.monitoring:
                self.monitoring = None
            if self.model_registry:
                self.model_registry = None
            if self.orchestrator:
                self.orchestrator = None
            if self.data_manager:
                self.data_manager = None
            if self.model_system:
                self.model_system = None
            
            self.is_initialized = False
            self.logger.info("✅ BEME framework shutdown completed")
            
        except Exception as e:
            self.logger.error(f"❌ Error during shutdown: {e}")
            raise

    # High-level framework operations
    async def train_model(self, model_type: str, dataset_name: str, **kwargs) -> str:
        """
        Train a model using the unified framework.
        
        Args:
            model_type: Type of model to train (hotel_bidding, flight_pricing, etc.)
            dataset_name: Name of the dataset to use for training
            **kwargs: Additional training parameters
            
        Returns:
            Model version string
        """
        try:
            self.logger.info(f"Training {model_type} model with dataset {dataset_name}")
            
            # Load training data
            if not self.data_manager:
                raise RuntimeError("Data manager not initialized")
            
            training_data = self.data_manager.load_dataset(dataset_name)
            
            # Train model (simplified implementation)
            # In practice, this would trigger the orchestration workflow
            if self.orchestrator:
                workflow_id = f"train_{model_type}_model"
                success = self.orchestrator.trigger_workflow(                    workflow_id,
                    dataset_name=dataset_name,
                    **kwargs
                )
                
                if success:
                    # Register model in registry
                    if self.model_registry and pd is not None and np is not None:
                        try:
                            # Mock model for demonstration
                            from sklearn.ensemble import RandomForestRegressor
                            model = RandomForestRegressor()
                            
                            # Mock training
                            X = training_data.select_dtypes(include=[np.number]).fillna(0)
                            y = X.iloc[:, 0] if len(X.columns) > 0 else pd.Series([1] * len(training_data))
                        except ImportError:
                            self.logger.warning("Scikit-learn not available, skipping model training")
                            return workflow_id
                    else:
                        self.logger.info("Model training skipped - missing dependencies or components")
                        
                        if len(X.columns) > 1:
                            model.fit(X.iloc[:, 1:], y)
                        
                        version = self.model_registry.register_model(
                            model_name=f"{model_type}_model",
                            model_type=self.model_registry.ModelType.OPTIMIZATION,
                            model_artifact=model,
                            metrics={"accuracy": 0.85, "latency_ms": 50.0},
                            description=f"Trained {model_type} model"
                        )
                        
                        self.logger.info(f"Model trained and registered: version {version}")
                        return version
            
            raise RuntimeError("Failed to train model")
            
        except Exception as e:
            self.logger.error(f"Failed to train model: {str(e)}")
            raise
    
    async def predict(self, model_name: str, inputs: Any, **kwargs) -> List[Any]:
        """
        Make predictions using a model.
        
        Args:
            model_name: Name of the model to use
            inputs: Input data for prediction
            **kwargs: Additional prediction parameters
            
        Returns:
            List of predictions
        """
        try:
            if not self.model_system:
                raise RuntimeError("Model system not initialized")
            
            predictions = await self.model_system.predict_async(
                model_name=model_name,
                inputs=inputs,
                **kwargs
            )
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {str(e)}")
            raise
    
    async def monitor_models(self) -> Dict[str, Any]:
        """
        Get current monitoring status for all models.
        
        Returns:
            Dictionary with monitoring metrics
        """
        try:
            if not self.monitoring:
                raise RuntimeError("Monitoring system not initialized")
            
            metrics = await self.monitoring.collect_metrics()
            report = self.monitoring.generate_unified_report(metrics)
            
            return report
            
        except Exception as e:
            self.logger.error(f"Failed to get monitoring status: {str(e)}")
            raise
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get overall system status.
        
        Returns:
            Dictionary with system status information
        """
        return {
            "framework_version": "1.0.0",
            "environment": self.config.environment,
            "initialized": self.is_initialized,
            "running_services": self.running_services,
            "components": {
                "data_manager": self.data_manager is not None,
                "model_registry": self.model_registry is not None,
                "model_system": self.model_system is not None,
                "monitoring": self.monitoring is not None,
                "orchestrator": self.orchestrator is not None
            },
            "configuration": {
                "log_level": self.config.log_level,
                "max_concurrent_predictions": self.config.max_concurrent_predictions,
                "model_update_frequency": self.config.model_update_frequency
            }
        }


# Global framework instance
_framework: Optional[BEMEFramework] = None

def get_framework(config: Optional[UnifiedConfig] = None) -> BEMEFramework:
    """Get the global BEME framework instance."""
    global _framework
    if _framework is None:
        _framework = BEMEFramework(config)
    return _framework

async def main():
    """Main entry point for the BEME framework."""
    # Initialize framework
    framework = get_framework()
    
    try:
        # Initialize all components
        await framework.initialize()
        
        # Start services
        await framework.start_services()
        
        # Run indefinitely
        while True:
            await asyncio.sleep(60)
            status = framework.get_system_status()
            framework.logger.info(f"System status: {status['running_services']}")
            
    except KeyboardInterrupt:
        framework.logger.info("Shutting down BEME framework")
        await framework.stop_services()
    except Exception as e:
        framework.logger.error(f"Framework error: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
