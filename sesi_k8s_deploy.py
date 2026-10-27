"""
=============================================================================
SESI KUBERNETES DEPLOYMENT GENERATOR
Multi-node distributed training + inference on K8s
=============================================================================

Generates production-ready K8s manifests:
  • Training Jobs (PyTorchJob via Kubeflow Training Operator)
  • Inference Deployments with GPU scheduling
  • PersistentVolumeClaims for datasets
  • HorizontalPodAutoscaler for inference
  • ConfigMaps / Secrets

HONEST SCOPE
────────────
• Requires a K8s cluster with NVIDIA GPU Operator for GPU workloads
• PyTorchJob requires Kubeflow Training Operator installed
• For Huawei Ascend, use volcano-sh + device plugin (different manifests)
=============================================================================
"""

from __future__ import annotations

import yaml
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger("sesi.k8s")

__all__ = ["K8sDeployConfig", "K8sManifestGenerator"]


@dataclass
class K8sDeployConfig:
    namespace: str = "sesi"
    image: str = "sesi/training:latest"
    
    # Training resources
    train_replicas: int = 4           # number of worker pods
    gpu_per_worker: int = 1
    gpu_type: str = "nvidia.com/gpu"  # or huawei.com/Ascend910
    cpu_per_worker: int = 8
    mem_per_worker: str = "32Gi"
    
    # Inference resources
    infer_replicas: int = 2
    infer_gpu: int = 1
    
    # Storage
    data_pvc_size: str = "500Gi"
    model_pvc_size: str = "100Gi"
    storage_class: str = "standard"
    
    # Training config
    backend: str = "nccl"             # nccl | gloo
    master_port: int = 29500


class K8sManifestGenerator:
    """Generate complete K8s manifests for SESI deployment."""
    
    def __init__(self, cfg: K8sDeployConfig):
        self.cfg = cfg
        self.output_dir = Path("./manifests")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_namespace(self) -> Dict:
        return {
            "apiVersion": "v1",
            "kind": "Namespace",
            "metadata": {"name": self.cfg.namespace}
        }
    
    def generate_data_pvc(self) -> Dict:
        """PVC for training datasets (TCGA, GISAID, PDB, OpenFOAM)."""
        return {
            "apiVersion": "v1",
            "kind": "PersistentVolumeClaim",
            "metadata": {
                "name": "sesi-data-pvc",
                "namespace": self.cfg.namespace,
            },
            "spec": {
                "accessModes": ["ReadWriteMany"],
                "storageClassName": self.cfg.storage_class,
                "resources": {
                    "requests": {"storage": self.cfg.data_pvc_size}
                }
            }
        }
    
    def generate_model_pvc(self) -> Dict:
        """PVC for model checkpoints."""
        return {
            "apiVersion": "v1",
            "kind": "PersistentVolumeClaim",
            "metadata": {
                "name": "sesi-model-pvc",
                "namespace": self.cfg.namespace,
            },
            "spec": {
                "accessModes": ["ReadWriteMany"],
                "storageClassName": self.cfg.storage_class,
                "resources": {
                    "requests": {"storage": self.cfg.model_pvc_size}
                }
            }
        }
    
    def generate_pytorch_job(self, job_name: str = "sesi-train",
                             script: str = "train.py") -> Dict:
        """
        Generate a PyTorchJob manifest (Kubeflow Training Operator).
        
        This handles multi-node DDP automatically via the operator.
        """
        return {
            "apiVersion": "kubeflow.org/v1",
            "kind": "PyTorchJob",
            "metadata": {
                "name": job_name,
                "namespace": self.cfg.namespace,
            },
            "spec": {
                "nprocPerNode": str(self.cfg.gpu_per_worker),
                "pytorchReplicaSpecs": {
                    "Master": {
                        "replicas": 1,
                        "restartPolicy": "OnFailure",
                        "template": {
                            "spec": {
                                "containers": [{
                                    "name": "pytorch",
                                    "image": self.cfg.image,
                                    "command": ["python", script,
                                               "--mode", "train",
                                               "--distributed"],
                                    "resources": {
                                        "limits": {
                                            self.cfg.gpu_type: self.cfg.gpu_per_worker,
                                            "cpu": self.cfg.cpu_per_worker,
                                            "memory": self.cfg.mem_per_worker,
                                        }
                                    },
                                    "env": [
                                        {"name": "NCCL_DEBUG", "value": "INFO"},
                                        {"name": "MASTER_PORT",
                                         "value": str(self.cfg.master_port)},
                                    ],
                                    "volumeMounts": [
                                        {"name": "data", "mountPath": "/data"},
                                        {"name": "models", "mountPath": "/models"},
                                    ]
                                }],
                                "volumes": [
                                    {"name": "data",
                                     "persistentVolumeClaim": {"claimName": "sesi-data-pvc"}},
                                    {"name": "models",
                                     "persistentVolumeClaim": {"claimName": "sesi-model-pvc"}},
                                ]
                            }
                        }
                    },
                    "Worker": {
                        "replicas": self.cfg.train_replicas,
                        "restartPolicy": "OnFailure",
                        "template": {
                            "spec": {
                                "containers": [{
                                    "name": "pytorch",
                                    "image": self.cfg.image,
                                    "command": ["python", script,
                                               "--mode", "train",
                                               "--distributed"],
                                    "resources": {
                                        "limits": {
                                            self.cfg.gpu_type: self.cfg.gpu_per_worker,
                                            "cpu": self.cfg.cpu_per_worker,
                                            "memory": self.cfg.mem_per_worker,
                                        }
                                    },
                                    "volumeMounts": [
                                        {"name": "data", "mountPath": "/data"},
                                        {"name": "models", "mountPath": "/models"},
                                    ]
                                }],
                                "volumes": [
                                    {"name": "data",
                                     "persistentVolumeClaim": {"claimName": "sesi-data-pvc"}},
                                    {"name": "models",
                                     "persistentVolumeClaim": {"claimName": "sesi-model-pvc"}},
                                ]
                            }
                        }
                    }
                }
            }
        }
    
    def generate_inference_deployment(self, name: str = "sesi-infer") -> Dict:
        """Deployment for inference serving."""
        return {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": name,
                "namespace": self.cfg.namespace,
            },
            "spec": {
                "replicas": self.cfg.infer_replicas,
                "selector": {"matchLabels": {"app": name}},
                "template": {
                    "metadata": {"labels": {"app": name}},
                    "spec": {
                        "containers": [{
                            "name": "inference",
                            "image": self.cfg.image,
                            "command": ["python", "serve.py"],
                            "ports": [{"containerPort": 8080}],
                            "resources": {
                                "limits": {
                                    self.cfg.gpu_type: self.cfg.infer_gpu,
                                }
                            },
                            "volumeMounts": [
                                {"name": "models", "mountPath": "/models"}
                            ]
                        }],
                        "volumes": [
                            {"name": "models",
                             "persistentVolumeClaim": {"claimName": "sesi-model-pvc"}}
                        ]
                    }
                }
            }
        }
    
    def generate_service(self) -> Dict:
        return {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": "sesi-infer-svc",
                "namespace": self.cfg.namespace,
            },
            "spec": {
                "selector": {"app": "sesi-infer"},
                "ports": [{"port": 80, "targetPort": 8080}],
                "type": "ClusterIP",
            }
        }
    
    def generate_hpa(self) -> Dict:
        """HorizontalPodAutoscaler for inference."""
        return {
            "apiVersion": "autoscaling/v2",
            "kind": "HorizontalPodAutoscaler",
            "metadata": {
                "name": "sesi-infer-hpa",
                "namespace": self.cfg.namespace,
            },
            "spec": {
                "scaleTargetRef": {
                    "apiVersion": "apps/v1",
                    "kind": "Deployment",
                    "name": "sesi-infer",
                },
                "minReplicas": 1,
                "maxReplicas": 10,
                "metrics": [{
                    "type": "Resource",
                    "resource": {
                        "name": "cpu",
                        "target": {"type": "Utilization", "averageUtilization": 70}
                    }
                }]
            }
        }
    
    def generate_all(self) -> Dict[str, Dict]:
        """Generate all manifests."""
        return {
            "00_namespace.yaml": self.generate_namespace(),
            "01_data_pvc.yaml": self.generate_data_pvc(),
            "02_model_pvc.yaml": self.generate_model_pvc(),
            "10_pytorch_job.yaml": self.generate_pytorch_job(),
            "20_inference_deployment.yaml": self.generate_inference_deployment(),
            "21_service.yaml": self.generate_service(),
            "22_hpa.yaml": self.generate_hpa(),
        }
    
    def write_all(self) -> List[Path]:
        """Write all manifests to disk."""
        manifests = self.generate_all()
        paths = []
        
        for filename, manifest in manifests.items():
            path = self.output_dir / filename
            with open(path, 'w') as f:
                yaml.dump(manifest, f, default_flow_style=False, sort_keys=False)
            paths.append(path)
            logger.info(f"Wrote {path}")
        
        return paths


def generate_dockerfile() -> str:
    """Generate Dockerfile for the SESI training image."""
    return """
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir \\
    requests numpy scipy \\
    biotite openmm openmm-ml \\
    pyyaml kubernetes \\
    tensorboard

# Copy application code
COPY . /app

# Default entrypoint
ENTRYPOINT ["python"]
CMD ["train.py", "--mode", "train"]
"""
