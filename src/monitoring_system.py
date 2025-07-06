# monitoring/monitoring_system.py
"""
P.O.C.E. Project Creator - Advanced Monitoring and Alerting System v4.0
Comprehensive monitoring with Prometheus metrics, Grafana dashboards,
alerting rules, health checks, and performance analytics
"""

import time
import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import threading
import queue
import statistics
from pathlib import Path
import os
import subprocess
import psutil
import socket
import urllib.parse

# Prometheus client imports
try:
    from prometheus_client import (
        Counter, Histogram, Gauge, Summary, Info,
        CollectorRegistry, generate_latest, start_http_server,
        multiprocess, CONTENT_TYPE_LATEST
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logging.warning("Prometheus client not available. Metrics collection disabled.")

# HTTP client for webhook notifications
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

logger = logging.getLogger(__name__)

# ==========================================
# MONITORING CONFIGURATION AND ENUMS
# ==========================================

class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class MetricType(Enum):
    """Types of metrics"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"

class HealthStatus(Enum):
    """Health check status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"

@dataclass
class AlertRule:
    """Alert rule configuration"""
    name: str
    description: str
    metric_name: str
    condition: str  # e.g., "> 0.8", "< 100"
    severity: AlertSeverity
    duration: int = 300  # seconds
    enabled: bool = True
    notification_channels: List[str] = field(default_factory=list)

@dataclass
class Alert:
    """Alert instance"""
    alert_id: str
    rule_name: str
    severity: AlertSeverity
    message: str
    timestamp: datetime
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class HealthCheck:
    """Health check configuration"""
    name: str
    check_function: Callable[[], bool]
    interval_seconds: int = 30
    timeout_seconds: int = 10
    critical: bool = True
    enabled: bool = True

# ==========================================
# METRICS COLLECTION SYSTEM
# ==========================================

class MetricsCollector:
    """Collects and manages application metrics"""
    
    def __init__(self):
        self.registry = CollectorRegistry() if PROMETHEUS_AVAILABLE else None
        self.metrics: Dict[str, Any] = {}
        self.custom_metrics: Dict[str, Dict] = {}
        
        if PROMETHEUS_AVAILABLE:
            self._initialize_default_metrics()
    
    def _initialize_default_metrics(self):
        """Initialize default application metrics"""
        # Application metrics
        self.metrics['app_requests_total'] = Counter(
            'poce_app_requests_total',
            'Total number of application requests',
            ['method', 'endpoint', 'status'],
            registry=self.registry
        )
        
        self.metrics['app_request_duration'] = Histogram(
            'poce_app_request_duration_seconds',
            'Application request duration',
            ['method', 'endpoint'],
            registry=self.registry
        )
        
        self.metrics['app_active_users'] = Gauge(
            'poce_app_active_users',
            'Number of active users',
            registry=self.registry
        )
        
        # Workflow metrics
        self.metrics['workflow_executions_total'] = Counter(
            'poce_workflow_executions_total',
            'Total workflow executions',
            ['workflow_type', 'status'],
            registry=self.registry
        )
        
        self.metrics['workflow_duration'] = Histogram(
            'poce_workflow_duration_seconds',
            'Workflow execution duration',
            ['workflow_type'],
            registry=self.registry
        )
        
        self.metrics['workflow_queue_size'] = Gauge(
            'poce_workflow_queue_size',
            'Number of workflows in queue',
            registry=self.registry
        )
        
        # MCP Server metrics
        self.metrics['mcp_server_requests'] = Counter(
            'poce_mcp_server_requests_total',
            'MCP server requests',
            ['server_name', 'operation', 'status'],
            registry=self.registry
        )
        
        self.metrics['mcp_server_response_time'] = Histogram(
            'poce_mcp_server_response_time_seconds',
            'MCP server response time',
            ['server_name', 'operation'],
            registry=self.registry
        )
        
        self.metrics['mcp_server_health'] = Gauge(
            'poce_mcp_server_health',
            'MCP server health status (1=healthy, 0=unhealthy)',
            ['server_name'],
            registry=self.registry
        )
        
        # System metrics
        self.metrics['system_cpu_usage'] = Gauge(
            'poce_system_cpu_usage_percent',
            'System CPU usage percentage',
            registry=self.registry
        )
        
        self.metrics['system_memory_usage'] = Gauge(
            'poce_system_memory_usage_bytes',
            'System memory usage in bytes',
            registry=self.registry
        )
        
        self.metrics['system_disk_usage'] = Gauge(
            'poce_system_disk_usage_percent',
            'System disk usage percentage',
            ['device'],
            registry=self.registry
        )
        
        # Error metrics
        self.metrics['errors_total'] = Counter(
            'poce_errors_total',
            'Total number of errors',
            ['component', 'error_type'],
            registry=self.registry
        )
        
        # Performance metrics
        self.metrics['cache_hits'] = Counter(
            'poce_cache_hits_total',
            'Total cache hits',
            ['cache_type'],
            registry=self.registry
        )
        
        self.metrics['cache_misses'] = Counter(
            'poce_cache_misses_total',
            'Total cache misses',
            ['cache_type'],
            registry=self.registry
        )
    
    def increment_counter(self, metric_name: str, labels: Dict[str, str] = None, value: float = 1):
        """Increment a counter metric"""
        if not PROMETHEUS_AVAILABLE or metric_name not in self.metrics:
            return
        
        metric = self.metrics[metric_name]
        if labels:
            metric.labels(**labels).inc(value)
        else:
            metric.inc(value)
    
    def set_gauge(self, metric_name: str, value: float, labels: Dict[str, str] = None):
        """Set a gauge metric value"""
        if not PROMETHEUS_AVAILABLE or metric_name not in self.metrics:
            return
        
        metric = self.metrics[metric_name]
        if labels:
            metric.labels(**labels).set(value)
        else:
            metric.set(value)
    
    def time_histogram(self, metric_name: str, labels: Dict[str, str] = None):
        """Context manager for timing histogram metrics"""
        if not PROMETHEUS_AVAILABLE or metric_name not in self.metrics:
            return _NoOpTimer()
        
        metric = self.metrics[metric_name]
        if labels:
            return metric.labels(**labels).time()
        else:
            return metric.time()
    
    def record_histogram(self, metric_name: str, value: float, labels: Dict[str, str] = None):
        """Record a histogram value"""
        if not PROMETHEUS_AVAILABLE or metric_name not in self.metrics:
            return
        
        metric = self.metrics[metric_name]
        if labels:
            metric.labels(**labels).observe(value)
        else:
            metric.observe(value)
    
    def get_metrics_output(self) -> str:
        """Get Prometheus formatted metrics output"""
        if not PROMETHEUS_AVAILABLE:
            return "# Prometheus metrics not available\n"
        
        return generate_latest(self.registry).decode('utf-8')
    
    def register_custom_metric(self, name: str, metric_type: MetricType, 
                             description: str, labels: List[str] = None):
        """Register a custom metric"""
        if not PROMETHEUS_AVAILABLE:
            return
        
        labels = labels or []
        
        if metric_type == MetricType.COUNTER:
            metric = Counter(name, description, labels, registry=self.registry)
        elif metric_type == MetricType.GAUGE:
            metric = Gauge(name, description, labels, registry=self.registry)
        elif metric_type == MetricType.HISTOGRAM:
            metric = Histogram(name, description, labels, registry=self.registry)
        elif metric_type == MetricType.SUMMARY:
            metric = Summary(name, description, labels, registry=self.registry)
        else:
            raise ValueError(f"Unsupported metric type: {metric_type}")
        
        self.metrics[name] = metric
        self.custom_metrics[name] = {
            'type': metric_type,
            'description': description,
            'labels': labels
        }

class _NoOpTimer:
    """No-op timer for when Prometheus is not available"""
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        pass

# ==========================================
# SYSTEM MONITORING
# ==========================================

class SystemMonitor:
    """Monitors system resources and performance"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
        self.monitoring = True
        self.monitor_thread: Optional[threading.Thread] = None
        self.update_interval = 30  # seconds
    
    def start_monitoring(self):
        """Start system monitoring in background thread"""
        if self.monitor_thread and self.monitor_thread.is_alive():
            logger.warning("System monitoring already running")
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("System monitoring started")
    
    def stop_monitoring(self):
        """Stop system monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("System monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.monitoring:
            try:
                self._collect_cpu_metrics()
                self._collect_memory_metrics()
                self._collect_disk_metrics()
                self._collect_network_metrics()
                self._collect_process_metrics()
                
                time.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"Error in system monitoring: {e}")
                time.sleep(self.update_interval)
    
    def _collect_cpu_metrics(self):
        """Collect CPU usage metrics"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            self.metrics.set_gauge('system_cpu_usage', cpu_percent)
            
            # Per-CPU metrics
            cpu_percents = psutil.cpu_percent(interval=1, percpu=True)
            for i, cpu_pct in enumerate(cpu_percents):
                self.metrics.set_gauge('system_cpu_usage', cpu_pct, {'cpu': str(i)})
                
        except Exception as e:
            logger.error(f"Failed to collect CPU metrics: {e}")
    
    def _collect_memory_metrics(self):
        """Collect memory usage metrics"""
        try:
            memory = psutil.virtual_memory()
            self.metrics.set_gauge('system_memory_usage', memory.used)
            
            # Additional memory metrics
            if 'system_memory_total' not in self.metrics.metrics:
                self.metrics.register_custom_metric(
                    'system_memory_total', MetricType.GAUGE, 'Total system memory'
                )
                self.metrics.register_custom_metric(
                    'system_memory_available', MetricType.GAUGE, 'Available system memory'
                )
                self.metrics.register_custom_metric(
                    'system_memory_percent', MetricType.GAUGE, 'Memory usage percentage'
                )
            
            self.metrics.set_gauge('system_memory_total', memory.total)
            self.metrics.set_gauge('system_memory_available', memory.available)
            self.metrics.set_gauge('system_memory_percent', memory.percent)
            
        except Exception as e:
            logger.error(f"Failed to collect memory metrics: {e}")
    
    def _collect_disk_metrics(self):
        """Collect disk usage metrics"""
        try:
            # Get disk usage for all mounted filesystems
            partitions = psutil.disk_partitions()
            
            for partition in partitions:
                try:
                    disk_usage = psutil.disk_usage(partition.mountpoint)
                    usage_percent = (disk_usage.used / disk_usage.total) * 100
                    
                    self.metrics.set_gauge(
                        'system_disk_usage',
                        usage_percent,
                        {'device': partition.device, 'mountpoint': partition.mountpoint}
                    )
                    
                except (PermissionError, FileNotFoundError):
                    # Skip inaccessible filesystems
                    continue
                    
        except Exception as e:
            logger.error(f"Failed to collect disk metrics: {e}")
    
    def _collect_network_metrics(self):
        """Collect network metrics"""
        try:
            if 'system_network_bytes_sent' not in self.metrics.metrics:
                self.metrics.register_custom_metric(
                    'system_network_bytes_sent', MetricType.COUNTER, 'Network bytes sent'
                )
                self.metrics.register_custom_metric(
                    'system_network_bytes_recv', MetricType.COUNTER, 'Network bytes received'
                )
            
            net_io = psutil.net_io_counters()
            self.metrics.set_gauge('system_network_bytes_sent', net_io.bytes_sent)
            self.metrics.set_gauge('system_network_bytes_recv', net_io.bytes_recv)
            
        except Exception as e:
            logger.error(f"Failed to collect network metrics: {e}")
    
    def _collect_process_metrics(self):
        """Collect process-specific metrics"""
        try:
            process = psutil.Process()
            
            if 'process_cpu_percent' not in self.metrics.metrics:
                self.metrics.register_custom_metric(
                    'process_cpu_percent', MetricType.GAUGE, 'Process CPU usage percentage'
                )
                self.metrics.register_custom_metric(
                    'process_memory_rss', MetricType.GAUGE, 'Process RSS memory'
                )
                self.metrics.register_custom_metric(
                    'process_memory_vms', MetricType.GAUGE, 'Process VMS memory'
                )
                self.metrics.register_custom_metric(
                    'process_open_fds', MetricType.GAUGE, 'Process open file descriptors'
                )
                self.metrics.register_custom_metric(
                    'process_threads', MetricType.GAUGE, 'Process thread count'
                )
            
            # Process metrics
            self.metrics.set_gauge('process_cpu_percent', process.cpu_percent())
            
            memory_info = process.memory_info()
            self.metrics.set_gauge('process_memory_rss', memory_info.rss)
            self.metrics.set_gauge('process_memory_vms', memory_info.vms)
            
            try:
                self.metrics.set_gauge('process_open_fds', process.num_fds())
            except AttributeError:
                # num_fds() not available on Windows
                pass
            
            self.metrics.set_gauge('process_threads', process.num_threads())
            
        except Exception as e:
            logger.error(f"Failed to collect process metrics: {e}")

# ==========================================
# HEALTH CHECK SYSTEM
# ==========================================

class HealthCheckManager:
    """Manages application health checks"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
        self.health_checks: Dict[str, HealthCheck] = {}
        self.health_status: Dict[str, Dict] = {}
        self.check_threads: Dict[str, threading.Thread] = {}
        self.running = False
        
        # Register health check metrics
        if PROMETHEUS_AVAILABLE:
            self.metrics.register_custom_metric(
                'health_check_status', MetricType.GAUGE,
                'Health check status (1=healthy, 0=unhealthy)', ['check_name']
            )
            self.metrics.register_custom_metric(
                'health_check_duration', MetricType.HISTOGRAM,
                'Health check duration', ['check_name']
            )
    
    def register_health_check(self, health_check: HealthCheck):
        """Register a new health check"""
        self.health_checks[health_check.name] = health_check
        self.health_status[health_check.name] = {
            'status': HealthStatus.UNKNOWN,
            'last_check': None,
            'error': None,
            'consecutive_failures': 0
        }
        
        logger.info(f"Registered health check: {health_check.name}")
    
    def start_health_checks(self):
        """Start all health checks"""
        if self.running:
            logger.warning("Health checks already running")
            return
        
        self.running = True
        
        for name, health_check in self.health_checks.items():
            if health_check.enabled:
                thread = threading.Thread(
                    target=self._health_check_loop,
                    args=(name, health_check),
                    daemon=True
                )
                thread.start()
                self.check_threads[name] = thread
        
        logger.info(f"Started {len(self.check_threads)} health check threads")
    
    def stop_health_checks(self):
        """Stop all health checks"""
        self.running = False
        
        # Wait for threads to finish
        for thread in self.check_threads.values():
            thread.join(timeout=5)
        
        self.check_threads.clear()
        logger.info("Stopped all health checks")
    
    def _health_check_loop(self, name: str, health_check: HealthCheck):
        """Health check execution loop"""
        while self.running:
            try:
                start_time = time.time()
                
                # Execute health check with timeout
                try:
                    result = self._execute_with_timeout(
                        health_check.check_function,
                        health_check.timeout_seconds
                    )
                    
                    duration = time.time() - start_time
                    
                    if result:
                        self._update_health_status(name, HealthStatus.HEALTHY, None)
                        self.metrics.set_gauge('health_check_status', 1, {'check_name': name})
                    else:
                        self._update_health_status(name, HealthStatus.UNHEALTHY, "Check returned False")
                        self.metrics.set_gauge('health_check_status', 0, {'check_name': name})
                    
                    self.metrics.record_histogram('health_check_duration', duration, {'check_name': name})
                    
                except TimeoutError:
                    self._update_health_status(name, HealthStatus.UNHEALTHY, "Timeout")
                    self.metrics.set_gauge('health_check_status', 0, {'check_name': name})
                
                except Exception as e:
                    self._update_health_status(name, HealthStatus.UNHEALTHY, str(e))
                    self.metrics.set_gauge('health_check_status', 0, {'check_name': name})
                
                time.sleep(health_check.interval_seconds)
                
            except Exception as e:
                logger.error(f"Error in health check loop for {name}: {e}")
                time.sleep(health_check.interval_seconds)
    
    def _execute_with_timeout(self, func: Callable, timeout: int):
        """Execute function with timeout"""
        result_queue = queue.Queue()
        
        def target():
            try:
                result = func()
                result_queue.put(('success', result))
            except Exception as e:
                result_queue.put(('error', e))
        
        thread = threading.Thread(target=target)
        thread.start()
        thread.join(timeout)
        
        if thread.is_alive():
            # Force thread termination is not directly possible,
            # but we can abandon it and raise timeout
            raise TimeoutError(f"Health check timed out after {timeout} seconds")
        
        try:
            status, result = result_queue.get_nowait()
            if status == 'error':
                raise result
            return result
        except queue.Empty:
            raise TimeoutError("No result received")
    
    def _update_health_status(self, name: str, status: HealthStatus, error: Optional[str]):
        """Update health check status"""
        previous_status = self.health_status[name]['status']
        
        self.health_status[name].update({
            'status': status,
            'last_check': datetime.utcnow(),
            'error': error
        })
        
        if status == HealthStatus.UNHEALTHY:
            self.health_status[name]['consecutive_failures'] += 1
        else:
            self.health_status[name]['consecutive_failures'] = 0
        
        # Log status changes
        if previous_status != status:
            logger.info(f"Health check {name} status changed: {previous_status.value} -> {status.value}")
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get overall health status"""
        overall_status = HealthStatus.HEALTHY
        unhealthy_checks = []
        degraded_checks = []
        
        for name, status_info in self.health_status.items():
            check = self.health_checks.get(name)
            current_status = status_info['status']
            
            if current_status == HealthStatus.UNHEALTHY:
                if check and check.critical:
                    overall_status = HealthStatus.UNHEALTHY
                    unhealthy_checks.append(name)
                else:
                    if overall_status == HealthStatus.HEALTHY:
                        overall_status = HealthStatus.DEGRADED
                    degraded_checks.append(name)
            
            elif current_status == HealthStatus.DEGRADED:
                if overall_status == HealthStatus.HEALTHY:
                    overall_status = HealthStatus.DEGRADED
                degraded_checks.append(name)
        
        return {
            'status': overall_status.value,
            'timestamp': datetime.utcnow().isoformat(),
            'checks': {
                name: {
                    'status': info['status'].value,
                    'last_check': info['last_check'].isoformat() if info['last_check'] else None,
                    'error': info['error'],
                    'consecutive_failures': info['consecutive_failures']
                }
                for name, info in self.health_status.items()
            },
            'summary': {
                'total_checks': len(self.health_checks),
                'healthy_checks': sum(1 for info in self.health_status.values() 
                                    if info['status'] == HealthStatus.HEALTHY),
                'unhealthy_checks': unhealthy_checks,
                'degraded_checks': degraded_checks
            }
        }

# ==========================================
# ALERTING SYSTEM
# ==========================================

class AlertManager:
    """Manages alerts and notifications"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.notification_channels: Dict[str, Callable] = {}
        self.evaluator_thread: Optional[threading.Thread] = None
        self.running = False
        self.evaluation_interval = 30  # seconds
    
    def register_alert_rule(self, rule: AlertRule):
        """Register an alert rule"""
        self.alert_rules[rule.name] = rule
        logger.info(f"Registered alert rule: {rule.name}")
    
    def register_notification_channel(self, name: str, handler: Callable[[Alert], None]):
        """Register a notification channel"""
        self.notification_channels[name] = handler
        logger.info(f"Registered notification channel: {name}")
    
    def start_alerting(self):
        """Start alert evaluation"""
        if self.running:
            logger.warning("Alert manager already running")
            return
        
        self.running = True
        self.evaluator_thread = threading.Thread(target=self._evaluation_loop, daemon=True)
        self.evaluator_thread.start()
        logger.info("Alert manager started")
    
    def stop_alerting(self):
        """Stop alert evaluation"""
        self.running = False
        if self.evaluator_thread:
            self.evaluator_thread.join(timeout=5)
        logger.info("Alert manager stopped")
    
    def _evaluation_loop(self):
        """Main alert evaluation loop"""
        while self.running:
            try:
                for rule_name, rule in self.alert_rules.items():
                    if rule.enabled:
                        self._evaluate_rule(rule)
                
                time.sleep(self.evaluation_interval)
                
            except Exception as e:
                logger.error(f"Error in alert evaluation: {e}")
                time.sleep(self.evaluation_interval)
    
    def _evaluate_rule(self, rule: AlertRule):
        """Evaluate a single alert rule"""
        try:
            # Get current metric value
            metric_value = self._get_metric_value(rule.metric_name)
            
            if metric_value is None:
                return
            
            # Evaluate condition
            condition_met = self._evaluate_condition(metric_value, rule.condition)
            
            alert_id = f"{rule.name}_{rule.metric_name}"
            
            if condition_met:
                if alert_id not in self.active_alerts:
                    # Create new alert
                    alert = Alert(
                        alert_id=alert_id,
                        rule_name=rule.name,
                        severity=rule.severity,
                        message=f"{rule.description} - Current value: {metric_value}",
                        timestamp=datetime.utcnow(),
                        metadata={
                            'metric_name': rule.metric_name,
                            'metric_value': metric_value,
                            'condition': rule.condition
                        }
                    )
                    
                    self.active_alerts[alert_id] = alert
                    self.alert_history.append(alert)
                    
                    # Send notifications
                    self._send_notifications(alert, rule.notification_channels)
                    
                    logger.warning(f"Alert triggered: {rule.name}")
            
            else:
                if alert_id in self.active_alerts:
                    # Resolve alert
                    alert = self.active_alerts[alert_id]
                    alert.resolved = True
                    alert.resolved_at = datetime.utcnow()
                    
                    del self.active_alerts[alert_id]
                    
                    # Send resolution notification
                    self._send_resolution_notification(alert, rule.notification_channels)
                    
                    logger.info(f"Alert resolved: {rule.name}")
        
        except Exception as e:
            logger.error(f"Error evaluating rule {rule.name}: {e}")
    
    def _get_metric_value(self, metric_name: str) -> Optional[float]:
        """Get current value of a metric"""
        # This is a simplified implementation
        # In a real system, you would query Prometheus or the metrics collector
        
        if not PROMETHEUS_AVAILABLE:
            return None
        
        # For demonstration, return a mock value
        # In reality, you would query the actual metric value
        import random
        return random.uniform(0, 100)
    
    def _evaluate_condition(self, value: float, condition: str) -> bool:
        """Evaluate alert condition"""
        try:
            # Simple condition evaluation
            # Support operators: >, <, >=, <=, ==, !=
            
            condition = condition.strip()
            
            for op in ['>=', '<=', '==', '!=', '>', '<']:
                if op in condition:
                    threshold = float(condition.split(op)[1].strip())
                    
                    if op == '>':
                        return value > threshold
                    elif op == '<':
                        return value < threshold
                    elif op == '>=':
                        return value >= threshold
                    elif op == '<=':
                        return value <= threshold
                    elif op == '==':
                        return value == threshold
                    elif op == '!=':
                        return value != threshold
            
            return False
            
        except Exception as e:
            logger.error(f"Error evaluating condition '{condition}': {e}")
            return False
    
    def _send_notifications(self, alert: Alert, channels: List[str]):
        """Send alert notifications"""
        for channel_name in channels:
            if channel_name in self.notification_channels:
                try:
                    self.notification_channels[channel_name](alert)
                except Exception as e:
                    logger.error(f"Failed to send notification via {channel_name}: {e}")
    
    def _send_resolution_notification(self, alert: Alert, channels: List[str]):
        """Send alert resolution notifications"""
        resolution_alert = Alert(
            alert_id=alert.alert_id + "_resolved",
            rule_name=alert.rule_name,
            severity=AlertSeverity.INFO,
            message=f"RESOLVED: {alert.message}",
            timestamp=datetime.utcnow(),
            metadata=alert.metadata
        )
        
        self._send_notifications(resolution_alert, channels)
    
    def get_alerts_summary(self) -> Dict[str, Any]:
        """Get alerts summary"""
        return {
            'active_alerts': len(self.active_alerts),
            'total_rules': len(self.alert_rules),
            'enabled_rules': sum(1 for rule in self.alert_rules.values() if rule.enabled),
            'alerts_by_severity': {
                severity.value: sum(1 for alert in self.active_alerts.values() 
                                  if alert.severity == severity)
                for severity in AlertSeverity
            },
            'recent_alerts': [
                {
                    'alert_id': alert.alert_id,
                    'rule_name': alert.rule_name,
                    'severity': alert.severity.value,
                    'message': alert.message,
                    'timestamp': alert.timestamp.isoformat(),
                    'resolved': alert.resolved
                }
                for alert in self.alert_history[-10:]  # Last 10 alerts
            ]
        }

# ==========================================
# NOTIFICATION HANDLERS
# ==========================================

class NotificationHandlers:
    """Collection of notification handlers"""
    
    @staticmethod
    def slack_webhook(webhook_url: str) -> Callable[[Alert], None]:
        """Create Slack webhook notification handler"""
        def handler(alert: Alert):
            if not REQUESTS_AVAILABLE:
                logger.warning("Requests library not available for Slack notifications")
                return
            
            color_map = {
                AlertSeverity.INFO: "good",
                AlertSeverity.WARNING: "warning", 
                AlertSeverity.CRITICAL: "danger",
                AlertSeverity.EMERGENCY: "danger"
            }
            
            payload = {
                "attachments": [
                    {
                        "color": color_map.get(alert.severity, "warning"),
                        "title": f"Alert: {alert.rule_name}",
                        "text": alert.message,
                        "fields": [
                            {
                                "title": "Severity",
                                "value": alert.severity.value.upper(),
                                "short": True
                            },
                            {
                                "title": "Timestamp",
                                "value": alert.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC"),
                                "short": True
                            }
                        ]
                    }
                ]
            }
            
            try:
                response = requests.post(webhook_url, json=payload, timeout=10)
                response.raise_for_status()
            except Exception as e:
                logger.error(f"Failed to send Slack notification: {e}")
        
        return handler
    
    @staticmethod
    def email_smtp(smtp_host: str, smtp_port: int, username: str, 
                   password: str, recipients: List[str]) -> Callable[[Alert], None]:
        """Create email SMTP notification handler"""
        def handler(alert: Alert):
            try:
                import smtplib
                from email.mime.text import MIMEText
                from email.mime.multipart import MIMEMultipart
                
                msg = MIMEMultipart()
                msg['From'] = username
                msg['To'] = ', '.join(recipients)
                msg['Subject'] = f"P.O.C.E Alert: {alert.rule_name} [{alert.severity.value.upper()}]"
                
                body = f"""
Alert Details:
- Rule: {alert.rule_name}
- Severity: {alert.severity.value.upper()}
- Message: {alert.message}
- Timestamp: {alert.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC")}
- Alert ID: {alert.alert_id}

Metadata:
{json.dumps(alert.metadata, indent=2)}
                """.strip()
                
                msg.attach(MIMEText(body, 'plain'))
                
                server = smtplib.SMTP(smtp_host, smtp_port)
                server.starttls()
                server.login(username, password)
                text = msg.as_string()
                server.sendmail(username, recipients, text)
                server.quit()
                
            except Exception as e:
                logger.error(f"Failed to send email notification: {e}")
        
        return handler
    
    @staticmethod
    def webhook_generic(webhook_url: str, headers: Dict[str, str] = None) -> Callable[[Alert], None]:
        """Create generic webhook notification handler"""
        def handler(alert: Alert):
            if not REQUESTS_AVAILABLE:
                logger.warning("Requests library not available for webhook notifications")
                return
            
            payload = {
                "alert_id": alert.alert_id,
                "rule_name": alert.rule_name,
                "severity": alert.severity.value,
                "message": alert.message,
                "timestamp": alert.timestamp.isoformat(),
                "resolved": alert.resolved,
                "metadata": alert.metadata
            }
            
            try:
                response = requests.post(
                    webhook_url, 
                    json=payload, 
                    headers=headers or {},
                    timeout=10
                )
                response.raise_for_status()
            except Exception as e:
                logger.error(f"Failed to send webhook notification: {e}")
        
        return handler

# ==========================================
# MAIN MONITORING SYSTEM
# ==========================================

class MonitoringSystem:
    """Main monitoring system coordinating all components"""
    
    def __init__(self):
        self.metrics = MetricsCollector()
        self.system_monitor = SystemMonitor(self.metrics)
        self.health_checks = HealthCheckManager(self.metrics)
        self.alerts = AlertManager(self.metrics)
        self.http_server_port = 9090
        self.http_server_thread: Optional[threading.Thread] = None
        
        # Initialize default health checks
        self._register_default_health_checks()
        
        # Initialize default alert rules
        self._register_default_alert_rules()
    
    def _register_default_health_checks(self):
        """Register default health checks"""
        # Database connectivity check
        def check_database():
            # Placeholder for database check
            return True
        
        # MCP server connectivity check
        def check_mcp_servers():
            # Placeholder for MCP server check
            return True
        
        # Disk space check
        def check_disk_space():
            try:
                disk_usage = psutil.disk_usage('/')
                usage_percent = (disk_usage.used / disk_usage.total) * 100
                return usage_percent < 90  # Alert if disk usage > 90%
            except:
                return False
        
        # Memory usage check
        def check_memory():
            try:
                memory = psutil.virtual_memory()
                return memory.percent < 85  # Alert if memory usage > 85%
            except:
                return False
        
        self.health_checks.register_health_check(HealthCheck(
            name="database_connectivity",
            check_function=check_database,
            interval_seconds=60,
            critical=True
        ))
        
        self.health_checks.register_health_check(HealthCheck(
            name="mcp_servers",
            check_function=check_mcp_servers,
            interval_seconds=30,
            critical=False
        ))
        
        self.health_checks.register_health_check(HealthCheck(
            name="disk_space",
            check_function=check_disk_space,
            interval_seconds=300,  # 5 minutes
            critical=True
        ))
        
        self.health_checks.register_health_check(HealthCheck(
            name="memory_usage",
            check_function=check_memory,
            interval_seconds=60,
            critical=False
        ))
    
    def _register_default_alert_rules(self):
        """Register default alert rules"""
        # CPU usage alert
        self.alerts.register_alert_rule(AlertRule(
            name="high_cpu_usage",
            description="High CPU usage detected",
            metric_name="system_cpu_usage",
            condition="> 80",
            severity=AlertSeverity.WARNING,
            duration=300,
            notification_channels=["slack", "email"]
        ))
        
        # Memory usage alert
        self.alerts.register_alert_rule(AlertRule(
            name="high_memory_usage",
            description="High memory usage detected",
            metric_name="system_memory_percent",
            condition="> 85",
            severity=AlertSeverity.CRITICAL,
            duration=180,
            notification_channels=["slack", "email"]
        ))
        
        # Disk usage alert
        self.alerts.register_alert_rule(AlertRule(
            name="high_disk_usage",
            description="High disk usage detected",
            metric_name="system_disk_usage",
            condition="> 90",
            severity=AlertSeverity.CRITICAL,
            duration=600,
            notification_channels=["slack", "email"]
        ))
        
        # Application error rate alert
        self.alerts.register_alert_rule(AlertRule(
            name="high_error_rate",
            description="High application error rate",
            metric_name="errors_total",
            condition="> 10",
            severity=AlertSeverity.WARNING,
            duration=300,
            notification_channels=["slack"]
        ))
    
    def start(self):
        """Start all monitoring components"""
        logger.info("Starting monitoring system...")
        
        # Start metrics HTTP server
        if PROMETHEUS_AVAILABLE:
            self._start_metrics_server()
        
        # Start system monitoring
        self.system_monitor.start_monitoring()
        
        # Start health checks
        self.health_checks.start_health_checks()
        
        # Start alerting
        self.alerts.start_alerting()
        
        logger.info("Monitoring system started successfully")
    
    def stop(self):
        """Stop all monitoring components"""
        logger.info("Stopping monitoring system...")
        
        # Stop alerting
        self.alerts.stop_alerting()
        
        # Stop health checks
        self.health_checks.stop_health_checks()
        
        # Stop system monitoring
        self.system_monitor.stop_monitoring()
        
        logger.info("Monitoring system stopped")
    
    def _start_metrics_server(self):
        """Start Prometheus metrics HTTP server"""
        if self.http_server_thread and self.http_server_thread.is_alive():
            logger.warning("Metrics server already running")
            return
        
        try:
            # Start HTTP server in background thread
            self.http_server_thread = threading.Thread(
                target=lambda: start_http_server(self.http_server_port, registry=self.metrics.registry),
                daemon=True
            )
            self.http_server_thread.start()
            logger.info(f"Metrics server started on port {self.http_server_port}")
            
        except Exception as e:
            logger.error(f"Failed to start metrics server: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'health': self.health_checks.get_health_status(),
            'alerts': self.alerts.get_alerts_summary(),
            'metrics_endpoint': f"http://localhost:{self.http_server_port}/metrics" if PROMETHEUS_AVAILABLE else None,
            'monitoring_components': {
                'system_monitor': self.system_monitor.monitoring,
                'health_checks': self.health_checks.running,
                'alert_manager': self.alerts.running,
                'metrics_server': self.http_server_thread.is_alive() if self.http_server_thread else False
            }
        }

# ==========================================
# EXAMPLE USAGE
# ==========================================

def example_monitoring_usage():
    """Example of how to use the monitoring system"""
    
    # Create and start monitoring system
    monitoring = MonitoringSystem()
    
    # Register notification channels
    monitoring.alerts.register_notification_channel(
        "slack",
        NotificationHandlers.slack_webhook("https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK")
    )
    
    monitoring.alerts.register_notification_channel(
        "email",
        NotificationHandlers.email_smtp(
            smtp_host="smtp.gmail.com",
            smtp_port=587,
            username="alerts@yourdomain.com",
            password="your_password",
            recipients=["admin@yourdomain.com"]
        )
    )
    
    # Start monitoring
    monitoring.start()
    
    # Simulate some metrics
    for i in range(10):
        monitoring.metrics.increment_counter('app_requests_total', {'method': 'GET', 'endpoint': '/api', 'status': '200'})
        monitoring.metrics.record_histogram('app_request_duration', 0.1 + i * 0.01, {'method': 'GET', 'endpoint': '/api'})
        time.sleep(1)
    
    # Get system status
    status = monitoring.get_system_status()
    print(f"System status: {json.dumps(status, indent=2)}")
    
    # Stop monitoring
    time.sleep(5)  # Let it run for a bit
    monitoring.stop()

if __name__ == "__main__":
    example_monitoring_usage()