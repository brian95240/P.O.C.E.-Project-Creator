# security/security_manager.py
"""
P.O.C.E. Project Creator - Enterprise Security Manager v4.0
Comprehensive security framework with encryption, authentication, monitoring,
and compliance features for enterprise DevOps environments
"""

import hashlib
import secrets
import base64
import json
import time
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import os
import re
import hmac
from pathlib import Path

# Cryptographic imports
try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa, padding
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.backends import default_backend
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    logging.warning("Cryptography library not available. Some security features disabled.")

# JWT imports (optional)
try:
    import jwt
    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False

logger = logging.getLogger(__name__)

# ==========================================
# SECURITY CONFIGURATION AND ENUMS
# ==========================================

class SecurityLevel(Enum):
    """Security levels for different environments"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    ENTERPRISE = "enterprise"

class EncryptionType(Enum):
    """Supported encryption types"""
    FERNET = "fernet"
    RSA = "rsa"
    AES = "aes"

class AuthenticationMethod(Enum):
    """Supported authentication methods"""
    TOKEN_BASED = "token"
    JWT = "jwt"
    API_KEY = "api_key"
    OAUTH2 = "oauth2"

@dataclass
class SecurityPolicy:
    """Security policy configuration"""
    security_level: SecurityLevel
    encryption_required: bool = True
    token_expiry_hours: int = 24
    max_login_attempts: int = 5
    password_min_length: int = 12
    require_mfa: bool = False
    audit_logging: bool = True
    data_retention_days: int = 90
    ip_whitelist: List[str] = field(default_factory=list)
    allowed_domains: List[str] = field(default_factory=list)

@dataclass
class SecurityEvent:
    """Security event for audit logging"""
    event_id: str
    event_type: str
    user_id: Optional[str]
    ip_address: Optional[str]
    timestamp: datetime
    severity: str
    description: str
    metadata: Dict[str, Any] = field(default_factory=dict)

# ==========================================
# ENCRYPTION AND CRYPTOGRAPHY
# ==========================================

class EncryptionManager:
    """Manages encryption and decryption operations"""
    
    def __init__(self, security_policy: SecurityPolicy):
        self.policy = security_policy
        self.encryption_key: Optional[bytes] = None
        self.rsa_private_key: Optional[Any] = None
        self.rsa_public_key: Optional[Any] = None
        
        if CRYPTO_AVAILABLE:
            self._initialize_encryption()
    
    def _initialize_encryption(self):
        """Initialize encryption keys"""
        try:
            # Generate or load Fernet key
            key_file = Path("security/encryption.key")
            if key_file.exists():
                with open(key_file, 'rb') as f:
                    self.encryption_key = f.read()
            else:
                self.encryption_key = Fernet.generate_key()
                key_file.parent.mkdir(exist_ok=True)
                with open(key_file, 'wb') as f:
                    f.write(self.encryption_key)
                os.chmod(key_file, 0o600)  # Restrict permissions
            
            # Generate RSA key pair for enterprise security
            if self.policy.security_level in [SecurityLevel.PRODUCTION, SecurityLevel.ENTERPRISE]:
                self._generate_rsa_keys()
                
        except Exception as e:
            logger.error(f"Failed to initialize encryption: {e}")
            if self.policy.security_level == SecurityLevel.ENTERPRISE:
                raise
    
    def _generate_rsa_keys(self):
        """Generate RSA key pair"""
        if not CRYPTO_AVAILABLE:
            return
        
        private_key_file = Path("security/rsa_private.pem")
        public_key_file = Path("security/rsa_public.pem")
        
        if private_key_file.exists() and public_key_file.exists():
            # Load existing keys
            with open(private_key_file, 'rb') as f:
                self.rsa_private_key = serialization.load_pem_private_key(
                    f.read(), password=None, backend=default_backend()
                )
            
            with open(public_key_file, 'rb') as f:
                self.rsa_public_key = serialization.load_pem_public_key(
                    f.read(), backend=default_backend()
                )
        else:
            # Generate new keys
            self.rsa_private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=4096,
                backend=default_backend()
            )
            self.rsa_public_key = self.rsa_private_key.public_key()
            
            # Save keys
            private_key_file.parent.mkdir(exist_ok=True)
            
            with open(private_key_file, 'wb') as f:
                f.write(self.rsa_private_key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.PKCS8,
                    encryption_algorithm=serialization.NoEncryption()
                ))
            os.chmod(private_key_file, 0o600)
            
            with open(public_key_file, 'wb') as f:
                f.write(self.rsa_public_key.public_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PublicFormat.SubjectPublicKeyInfo
                ))
    
    def encrypt_data(self, data: str, encryption_type: EncryptionType = EncryptionType.FERNET) -> str:
        """Encrypt sensitive data"""
        if not CRYPTO_AVAILABLE:
            logger.warning("Encryption not available, returning data as-is")
            return data
        
        try:
            if encryption_type == EncryptionType.FERNET and self.encryption_key:
                fernet = Fernet(self.encryption_key)
                encrypted_data = fernet.encrypt(data.encode())
                return base64.urlsafe_b64encode(encrypted_data).decode()
            
            elif encryption_type == EncryptionType.RSA and self.rsa_public_key:
                encrypted_data = self.rsa_public_key.encrypt(
                    data.encode(),
                    padding.OAEP(
                        mgf=padding.MGF1(algorithm=hashes.SHA256()),
                        algorithm=hashes.SHA256(),
                        label=None
                    )
                )
                return base64.urlsafe_b64encode(encrypted_data).decode()
            
            else:
                raise ValueError(f"Unsupported encryption type: {encryption_type}")
                
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            if self.policy.security_level == SecurityLevel.ENTERPRISE:
                raise
            return data
    
    def decrypt_data(self, encrypted_data: str, encryption_type: EncryptionType = EncryptionType.FERNET) -> str:
        """Decrypt sensitive data"""
        if not CRYPTO_AVAILABLE:
            logger.warning("Decryption not available, returning data as-is")
            return encrypted_data
        
        try:
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode())
            
            if encryption_type == EncryptionType.FERNET and self.encryption_key:
                fernet = Fernet(self.encryption_key)
                decrypted_data = fernet.decrypt(encrypted_bytes)
                return decrypted_data.decode()
            
            elif encryption_type == EncryptionType.RSA and self.rsa_private_key:
                decrypted_data = self.rsa_private_key.decrypt(
                    encrypted_bytes,
                    padding.OAEP(
                        mgf=padding.MGF1(algorithm=hashes.SHA256()),
                        algorithm=hashes.SHA256(),
                        label=None
                    )
                )
                return decrypted_data.decode()
            
            else:
                raise ValueError(f"Unsupported encryption type: {encryption_type}")
                
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            if self.policy.security_level == SecurityLevel.ENTERPRISE:
                raise
            return encrypted_data

# ==========================================
# AUTHENTICATION AND AUTHORIZATION
# ==========================================

class TokenManager:
    """Manages authentication tokens and API keys"""
    
    def __init__(self, security_policy: SecurityPolicy, encryption_manager: EncryptionManager):
        self.policy = security_policy
        self.encryption = encryption_manager
        self.active_tokens: Dict[str, Dict] = {}
        self.revoked_tokens: set = set()
    
    def generate_api_key(self, user_id: str, permissions: List[str] = None) -> str:
        """Generate secure API key"""
        # Generate cryptographically secure random key
        key_data = {
            'user_id': user_id,
            'permissions': permissions or [],
            'created_at': datetime.utcnow().isoformat(),
            'expires_at': (datetime.utcnow() + timedelta(hours=self.policy.token_expiry_hours)).isoformat()
        }
        
        # Create unique token
        token_id = secrets.token_urlsafe(32)
        
        # Store token metadata
        self.active_tokens[token_id] = key_data
        
        # Create secure token format
        api_key = f"poce_{base64.urlsafe_b64encode(token_id.encode()).decode().rstrip('=')}"
        
        return api_key
    
    def generate_jwt_token(self, user_id: str, permissions: List[str] = None) -> Optional[str]:
        """Generate JWT token"""
        if not JWT_AVAILABLE:
            logger.warning("JWT library not available")
            return None
        
        payload = {
            'user_id': user_id,
            'permissions': permissions or [],
            'iat': datetime.utcnow(),
            'exp': datetime.utcnow() + timedelta(hours=self.policy.token_expiry_hours),
            'iss': 'poce-project-creator',
            'aud': 'poce-api'
        }
        
        # Use encryption key as JWT secret
        secret = self.encryption.encryption_key or "fallback_secret"
        
        return jwt.encode(payload, secret, algorithm='HS256')
    
    def validate_token(self, token: str) -> Tuple[bool, Optional[Dict]]:
        """Validate authentication token"""
        try:
            # Check if token is revoked
            if token in self.revoked_tokens:
                return False, {'error': 'Token revoked'}
            
            # Handle API key format
            if token.startswith('poce_'):
                token_id = base64.urlsafe_b64decode(
                    token[5:] + '=' * (-len(token[5:]) % 4)
                ).decode()
                
                if token_id in self.active_tokens:
                    token_data = self.active_tokens[token_id]
                    
                    # Check expiration
                    expires_at = datetime.fromisoformat(token_data['expires_at'])
                    if datetime.utcnow() > expires_at:
                        return False, {'error': 'Token expired'}
                    
                    return True, token_data
                
                return False, {'error': 'Invalid token'}
            
            # Handle JWT token
            elif JWT_AVAILABLE:
                secret = self.encryption.encryption_key or "fallback_secret"
                payload = jwt.decode(token, secret, algorithms=['HS256'])
                return True, payload
            
            else:
                return False, {'error': 'Unsupported token format'}
                
        except Exception as e:
            logger.error(f"Token validation failed: {e}")
            return False, {'error': 'Token validation failed'}
    
    def revoke_token(self, token: str) -> bool:
        """Revoke authentication token"""
        try:
            self.revoked_tokens.add(token)
            
            # Remove from active tokens if API key
            if token.startswith('poce_'):
                token_id = base64.urlsafe_b64decode(
                    token[5:] + '=' * (-len(token[5:]) % 4)
                ).decode()
                
                if token_id in self.active_tokens:
                    del self.active_tokens[token_id]
            
            return True
            
        except Exception as e:
            logger.error(f"Token revocation failed: {e}")
            return False

# ==========================================
# AUDIT LOGGING AND MONITORING
# ==========================================

class SecurityAuditor:
    """Security audit logging and monitoring"""
    
    def __init__(self, security_policy: SecurityPolicy):
        self.policy = security_policy
        self.events: List[SecurityEvent] = []
        self.failed_attempts: Dict[str, List[datetime]] = {}
        
        # Setup audit log file
        self.audit_log_file = Path("security/audit.log")
        self.audit_log_file.parent.mkdir(exist_ok=True)
    
    def log_security_event(self, event_type: str, user_id: Optional[str] = None,
                          ip_address: Optional[str] = None, severity: str = "INFO",
                          description: str = "", metadata: Dict[str, Any] = None):
        """Log security event"""
        event = SecurityEvent(
            event_id=secrets.token_hex(16),
            event_type=event_type,
            user_id=user_id,
            ip_address=ip_address,
            timestamp=datetime.utcnow(),
            severity=severity,
            description=description,
            metadata=metadata or {}
        )
        
        self.events.append(event)
        
        # Write to audit log file
        if self.policy.audit_logging:
            self._write_audit_log(event)
        
        # Check for security violations
        self._check_security_violations(event)
    
    def _write_audit_log(self, event: SecurityEvent):
        """Write event to audit log file"""
        try:
            log_entry = {
                'event_id': event.event_id,
                'timestamp': event.timestamp.isoformat(),
                'event_type': event.event_type,
                'user_id': event.user_id,
                'ip_address': event.ip_address,
                'severity': event.severity,
                'description': event.description,
                'metadata': event.metadata
            }
            
            with open(self.audit_log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
                
        except Exception as e:
            logger.error(f"Failed to write audit log: {e}")
    
    def _check_security_violations(self, event: SecurityEvent):
        """Check for security violations and alert"""
        # Check for excessive failed login attempts
        if event.event_type == "authentication_failed" and event.ip_address:
            if event.ip_address not in self.failed_attempts:
                self.failed_attempts[event.ip_address] = []
            
            self.failed_attempts[event.ip_address].append(event.timestamp)
            
            # Remove old attempts (last hour)
            cutoff = datetime.utcnow() - timedelta(hours=1)
            self.failed_attempts[event.ip_address] = [
                attempt for attempt in self.failed_attempts[event.ip_address]
                if attempt > cutoff
            ]
            
            # Check if exceeded threshold
            if len(self.failed_attempts[event.ip_address]) > self.policy.max_login_attempts:
                self.log_security_event(
                    "security_violation",
                    ip_address=event.ip_address,
                    severity="CRITICAL",
                    description=f"Excessive failed login attempts from {event.ip_address}",
                    metadata={'attempt_count': len(self.failed_attempts[event.ip_address])}
                )
    
    def get_security_report(self, hours: int = 24) -> Dict[str, Any]:
        """Generate security report"""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        recent_events = [event for event in self.events if event.timestamp > cutoff]
        
        # Aggregate statistics
        event_types = {}
        severity_counts = {}
        ip_addresses = set()
        users = set()
        
        for event in recent_events:
            event_types[event.event_type] = event_types.get(event.event_type, 0) + 1
            severity_counts[event.severity] = severity_counts.get(event.severity, 0) + 1
            
            if event.ip_address:
                ip_addresses.add(event.ip_address)
            if event.user_id:
                users.add(event.user_id)
        
        return {
            'report_generated': datetime.utcnow().isoformat(),
            'time_period_hours': hours,
            'total_events': len(recent_events),
            'event_types': event_types,
            'severity_distribution': severity_counts,
            'unique_ip_addresses': len(ip_addresses),
            'unique_users': len(users),
            'top_ip_addresses': list(ip_addresses)[:10],
            'recent_critical_events': [
                {
                    'timestamp': event.timestamp.isoformat(),
                    'event_type': event.event_type,
                    'description': event.description
                }
                for event in recent_events
                if event.severity == "CRITICAL"
            ][-10:]  # Last 10 critical events
        }

# ==========================================
# INPUT VALIDATION AND SANITIZATION
# ==========================================

class InputValidator:
    """Validates and sanitizes input data"""
    
    def __init__(self, security_policy: SecurityPolicy):
        self.policy = security_policy
        
        # Dangerous patterns to detect
        self.sql_injection_patterns = [
            r"('|(\\x27)|(\\x2D)|(\\x22)|(\\x5C)|(-{2})|(\s*(or|OR)\s+))",
            r"(\s*(union|UNION)\s+)",
            r"(\s*(select|SELECT)\s+)",
            r"(\s*(insert|INSERT)\s+)",
            r"(\s*(delete|DELETE)\s+)",
            r"(\s*(update|UPDATE)\s+)",
            r"(\s*(drop|DROP)\s+)"
        ]
        
        self.xss_patterns = [
            r"<script[^>]*>.*?</script>",
            r"javascript:",
            r"vbscript:",
            r"onload\s*=",
            r"onerror\s*=",
            r"onclick\s*="
        ]
        
        self.path_traversal_patterns = [
            r"\.\./",
            r"\.\.\\",
            r"%2e%2e%2f",
            r"%2e%2e\\",
            r"..%2f",
            r"..%5c"
        ]
    
    def validate_project_name(self, name: str) -> Tuple[bool, str]:
        """Validate project name"""
        if not name:
            return False, "Project name cannot be empty"
        
        if len(name) > 100:
            return False, "Project name too long (max 100 characters)"
        
        # Check for invalid characters
        if not re.match(r"^[a-zA-Z0-9\-_\.]+$", name):
            return False, "Project name contains invalid characters"
        
        # Check for dangerous patterns
        if self._contains_dangerous_patterns(name):
            return False, "Project name contains potentially dangerous content"
        
        return True, "Valid"
    
    def validate_github_token(self, token: str) -> Tuple[bool, str]:
        """Validate GitHub token format"""
        if not token:
            return False, "GitHub token cannot be empty"
        
        # GitHub personal access tokens start with 'ghp_' or 'github_pat_'
        if not (token.startswith('ghp_') or token.startswith('github_pat_')):
            return False, "Invalid GitHub token format"
        
        if len(token) < 20:
            return False, "GitHub token too short"
        
        return True, "Valid"
    
    def sanitize_input(self, input_data: str, max_length: int = 1000) -> str:
        """Sanitize input data"""
        if not input_data:
            return ""
        
        # Truncate if too long
        sanitized = input_data[:max_length]
        
        # Remove null bytes
        sanitized = sanitized.replace('\x00', '')
        
        # Escape HTML characters
        html_escape_table = {
            "&": "&amp;",
            '"': "&quot;",
            "'": "&#x27;",
            ">": "&gt;",
            "<": "&lt;",
        }
        
        for char, escape in html_escape_table.items():
            sanitized = sanitized.replace(char, escape)
        
        return sanitized
    
    def _contains_dangerous_patterns(self, text: str) -> bool:
        """Check if text contains dangerous patterns"""
        text_lower = text.lower()
        
        # Check SQL injection patterns
        for pattern in self.sql_injection_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return True
        
        # Check XSS patterns
        for pattern in self.xss_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return True
        
        # Check path traversal patterns
        for pattern in self.path_traversal_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return True
        
        return False

# ==========================================
# SECURE CONFIGURATION MANAGER
# ==========================================

class SecureConfigManager:
    """Manages secure configuration with encryption"""
    
    def __init__(self, security_policy: SecurityPolicy):
        self.policy = security_policy
        self.encryption = EncryptionManager(security_policy)
        self.auditor = SecurityAuditor(security_policy)
        self.validator = InputValidator(security_policy)
        self.token_manager = TokenManager(security_policy, self.encryption)
        
        # Sensitive fields that should be encrypted
        self.sensitive_fields = {
            'github.token',
            'mcp_servers.api_keys',
            'database.password',
            'redis.password',
            'smtp.password'
        }
    
    def store_config(self, config: Dict[str, Any], config_path: str) -> bool:
        """Store configuration with encryption for sensitive data"""
        try:
            # Create a copy for modification
            secure_config = self._deep_copy_dict(config)
            
            # Encrypt sensitive fields
            self._encrypt_sensitive_fields(secure_config)
            
            # Save to file
            with open(config_path, 'w') as f:
                yaml.dump(secure_config, f, default_flow_style=False)
            
            # Set secure file permissions
            os.chmod(config_path, 0o600)
            
            # Log security event
            self.auditor.log_security_event(
                "config_updated",
                severity="INFO",
                description=f"Configuration updated: {config_path}"
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to store secure config: {e}")
            self.auditor.log_security_event(
                "config_error",
                severity="ERROR",
                description=f"Failed to store config: {e}"
            )
            return False
    
    def load_config(self, config_path: str) -> Optional[Dict[str, Any]]:
        """Load configuration with decryption"""
        try:
            with open(config_path, 'r') as f:
                secure_config = yaml.safe_load(f)
            
            # Decrypt sensitive fields
            self._decrypt_sensitive_fields(secure_config)
            
            # Log access
            self.auditor.log_security_event(
                "config_accessed",
                severity="INFO",
                description=f"Configuration loaded: {config_path}"
            )
            
            return secure_config
            
        except Exception as e:
            logger.error(f"Failed to load secure config: {e}")
            self.auditor.log_security_event(
                "config_error",
                severity="ERROR",
                description=f"Failed to load config: {e}"
            )
            return None
    
    def _encrypt_sensitive_fields(self, config: Dict[str, Any], path: str = ""):
        """Recursively encrypt sensitive fields"""
        for key, value in config.items():
            current_path = f"{path}.{key}" if path else key
            
            if isinstance(value, dict):
                self._encrypt_sensitive_fields(value, current_path)
            elif isinstance(value, str) and current_path in self.sensitive_fields:
                if value and not value.startswith("encrypted:"):
                    encrypted_value = self.encryption.encrypt_data(value)
                    config[key] = f"encrypted:{encrypted_value}"
    
    def _decrypt_sensitive_fields(self, config: Dict[str, Any], path: str = ""):
        """Recursively decrypt sensitive fields"""
        for key, value in config.items():
            current_path = f"{path}.{key}" if path else key
            
            if isinstance(value, dict):
                self._decrypt_sensitive_fields(value, current_path)
            elif isinstance(value, str) and value.startswith("encrypted:"):
                encrypted_value = value[10:]  # Remove "encrypted:" prefix
                decrypted_value = self.encryption.decrypt_data(encrypted_value)
                config[key] = decrypted_value
    
    def _deep_copy_dict(self, d: Dict) -> Dict:
        """Deep copy dictionary"""
        import copy
        return copy.deepcopy(d)

# ==========================================
# COMPLIANCE AND STANDARDS
# ==========================================

class ComplianceManager:
    """Manages compliance with security standards"""
    
    def __init__(self, security_policy: SecurityPolicy):
        self.policy = security_policy
        self.compliance_standards = {
            'SOC2': {
                'encryption_required': True,
                'audit_logging': True,
                'access_controls': True,
                'data_retention': True
            },
            'GDPR': {
                'data_encryption': True,
                'right_to_erasure': True,
                'consent_management': True,
                'breach_notification': True
            },
            'HIPAA': {
                'data_encryption': True,
                'access_logging': True,
                'user_authentication': True,
                'audit_trails': True
            }
        }
    
    def check_compliance(self, standard: str) -> Dict[str, Any]:
        """Check compliance with specific standard"""
        if standard not in self.compliance_standards:
            return {'compliant': False, 'error': f'Unknown standard: {standard}'}
        
        requirements = self.compliance_standards[standard]
        compliance_results = {}
        
        for requirement, required in requirements.items():
            if requirement == 'encryption_required':
                compliance_results[requirement] = self.policy.encryption_required
            elif requirement == 'audit_logging':
                compliance_results[requirement] = self.policy.audit_logging
            elif requirement == 'data_retention':
                compliance_results[requirement] = self.policy.data_retention_days > 0
            else:
                # Default to policy security level
                compliance_results[requirement] = (
                    self.policy.security_level in [SecurityLevel.PRODUCTION, SecurityLevel.ENTERPRISE]
                )
        
        # Calculate overall compliance
        compliant = all(compliance_results.values())
        
        return {
            'standard': standard,
            'compliant': compliant,
            'requirements': compliance_results,
            'checked_at': datetime.utcnow().isoformat()
        }

# ==========================================
# MAIN SECURITY MANAGER
# ==========================================

class EnterpriseSecurityManager:
    """Main security manager coordinating all security components"""
    
    def __init__(self, security_level: SecurityLevel = SecurityLevel.PRODUCTION):
        self.policy = SecurityPolicy(
            security_level=security_level,
            encryption_required=True,
            token_expiry_hours=24 if security_level != SecurityLevel.ENTERPRISE else 8,
            max_login_attempts=5,
            password_min_length=12 if security_level != SecurityLevel.ENTERPRISE else 16,
            require_mfa=security_level == SecurityLevel.ENTERPRISE,
            audit_logging=True,
            data_retention_days=90 if security_level != SecurityLevel.ENTERPRISE else 365
        )
        
        self.encryption = EncryptionManager(self.policy)
        self.auditor = SecurityAuditor(self.policy)
        self.validator = InputValidator(self.policy)
        self.token_manager = TokenManager(self.policy, self.encryption)
        self.config_manager = SecureConfigManager(self.policy)
        self.compliance = ComplianceManager(self.policy)
        
        # Initialize security
        self._initialize_security()
    
    def _initialize_security(self):
        """Initialize security subsystems"""
        logger.info(f"Initializing security with level: {self.policy.security_level.value}")
        
        # Log security initialization
        self.auditor.log_security_event(
            "security_initialized",
            severity="INFO",
            description=f"Security manager initialized with level: {self.policy.security_level.value}",
            metadata={'policy': self.policy.__dict__}
        )
    
    def authenticate_request(self, token: str, ip_address: str = None) -> Tuple[bool, Optional[Dict]]:
        """Authenticate incoming request"""
        # Validate IP address if whitelist is configured
        if self.policy.ip_whitelist and ip_address:
            if ip_address not in self.policy.ip_whitelist:
                self.auditor.log_security_event(
                    "authentication_failed",
                    ip_address=ip_address,
                    severity="WARNING",
                    description=f"Request from non-whitelisted IP: {ip_address}"
                )
                return False, {'error': 'IP address not whitelisted'}
        
        # Validate token
        valid, token_data = self.token_manager.validate_token(token)
        
        if valid:
            self.auditor.log_security_event(
                "authentication_success",
                user_id=token_data.get('user_id'),
                ip_address=ip_address,
                severity="INFO",
                description="Successful authentication"
            )
            return True, token_data
        else:
            self.auditor.log_security_event(
                "authentication_failed",
                ip_address=ip_address,
                severity="WARNING",
                description=f"Authentication failed: {token_data.get('error', 'Unknown error')}"
            )
            return False, token_data
    
    def generate_secure_token(self, user_id: str, permissions: List[str] = None) -> str:
        """Generate secure authentication token"""
        if JWT_AVAILABLE and self.policy.security_level in [SecurityLevel.PRODUCTION, SecurityLevel.ENTERPRISE]:
            token = self.token_manager.generate_jwt_token(user_id, permissions)
        else:
            token = self.token_manager.generate_api_key(user_id, permissions)
        
        self.auditor.log_security_event(
            "token_generated",
            user_id=user_id,
            severity="INFO",
            description="Authentication token generated",
            metadata={'permissions': permissions or []}
        )
        
        return token
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get comprehensive security status"""
        return {
            'security_level': self.policy.security_level.value,
            'encryption_enabled': CRYPTO_AVAILABLE and self.policy.encryption_required,
            'audit_logging': self.policy.audit_logging,
            'active_tokens': len(self.token_manager.active_tokens),
            'revoked_tokens': len(self.token_manager.revoked_tokens),
            'recent_events': len([e for e in self.auditor.events 
                                if e.timestamp > datetime.utcnow() - timedelta(hours=24)]),
            'compliance_status': {
                standard: self.compliance.check_compliance(standard)['compliant']
                for standard in self.compliance.compliance_standards.keys()
            },
            'last_updated': datetime.utcnow().isoformat()
        }

# ==========================================
# EXAMPLE USAGE AND TESTING
# ==========================================

def example_security_usage():
    """Example of how to use the security framework"""
    
    # Initialize security manager
    security = EnterpriseSecurityManager(SecurityLevel.ENTERPRISE)
    
    # Generate secure token
    token = security.generate_secure_token("user123", ["read", "write"])
    print(f"Generated token: {token[:20]}...")
    
    # Authenticate request
    valid, user_data = security.authenticate_request(token, "192.168.1.100")
    print(f"Authentication result: {valid}")
    
    # Check security status
    status = security.get_security_status()
    print(f"Security status: {status}")
    
    # Check compliance
    soc2_compliance = security.compliance.check_compliance("SOC2")
    print(f"SOC2 Compliance: {soc2_compliance['compliant']}")
    
    # Generate security report
    report = security.auditor.get_security_report(24)
    print(f"Security events in last 24h: {report['total_events']}")

if __name__ == "__main__":
    example_security_usage()