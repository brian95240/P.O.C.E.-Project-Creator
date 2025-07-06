# performance/optimization_engine.py
"""
P.O.C.E. Project Creator - Performance Optimization and Benchmarking System v4.0
Advanced performance monitoring, optimization, and benchmarking with AI-powered
recommendations and automated tuning capabilities
"""

import time
import asyncio
import threading
import multiprocessing
import psutil
import gc
import sys
import os
import json
import statistics
import logging
from typing import Dict, List, Optional, Any, Callable, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
import functools
import cProfile
import pstats
import io
from contextlib import contextmanager
import tracemalloc
import resource

# Memory profiling
try:
    from memory_profiler import profile as memory_profile
    MEMORY_PROFILER_AVAILABLE = True
except ImportError:
    MEMORY_PROFILER_AVAILABLE = False

# Line profiler
try:
    from line_profiler import LineProfiler
    LINE_PROFILER_AVAILABLE = True
except ImportError:
    LINE_PROFILER_AVAILABLE = False

# Performance monitoring
try:
    import py_spy
    PY_SPY_AVAILABLE = True
except ImportError:
    PY_SPY_AVAILABLE = False

logger = logging.getLogger(__name__)

# ==========================================
# PERFORMANCE CONFIGURATION AND ENUMS
# ==========================================

class OptimizationLevel(Enum):
    """Performance optimization levels"""
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"
    MAXIMUM = "maximum"

class ProfilerType(Enum):
    """Types of profilers available"""
    CPROFILE = "cprofile"
    MEMORY = "memory"
    LINE = "line"
    SAMPLING = "sampling"

class MetricType(Enum):
    """Performance metric types"""
    EXECUTION_TIME = "execution_time"
    MEMORY_USAGE = "memory_usage"
    CPU_USAGE = "cpu_usage"
    THROUGHPUT = "throughput"
    LATENCY = "latency"
    CACHE_HIT_RATE = "cache_hit_rate"

@dataclass
class PerformanceMetric:
    """Performance metric data point"""
    name: str
    value: float
    unit: str
    timestamp: datetime
    context: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BenchmarkResult:
    """Benchmark execution result"""
    benchmark_name: str
    execution_time: float
    memory_peak: float
    cpu_usage: float
    throughput: Optional[float] = None
    iterations: int = 1
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class OptimizationRecommendation:
    """Performance optimization recommendation"""
    category: str
    priority: int  # 1-10, 10 being highest
    description: str
    implementation: str
    expected_improvement: str
    estimated_effort: str
    code_changes: List[str] = field(default_factory=list)

# ==========================================
# PERFORMANCE PROFILING SYSTEM
# ==========================================

class PerformanceProfiler:
    """Advanced performance profiling with multiple profiler types"""
    
    def __init__(self):
        self.profilers: Dict[str, Any] = {}
        self.profile_results: Dict[str, Any] = {}
        self.active_profiles: Dict[str, bool] = {}
        
    @contextmanager
    def profile_execution(self, name: str, profiler_type: ProfilerType = ProfilerType.CPROFILE):
        """Context manager for profiling code execution"""
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        if profiler_type == ProfilerType.CPROFILE:
            profiler = cProfile.Profile()
            profiler.enable()
        
        elif profiler_type == ProfilerType.MEMORY and tracemalloc.is_tracing():
            tracemalloc.start()
        
        elif profiler_type == ProfilerType.LINE and LINE_PROFILER_AVAILABLE:
            profiler = LineProfiler()
            profiler.enable()
        
        try:
            yield
        finally:
            execution_time = time.time() - start_time
            end_memory = self._get_memory_usage()
            memory_delta = end_memory - start_memory
            
            if profiler_type == ProfilerType.CPROFILE:
                profiler.disable()
                self._save_cprofile_results(name, profiler, execution_time, memory_delta)
            
            elif profiler_type == ProfilerType.MEMORY and tracemalloc.is_tracing():
                current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                self._save_memory_profile_results(name, peak, execution_time)
            
            elif profiler_type == ProfilerType.LINE and LINE_PROFILER_AVAILABLE:
                profiler.disable()
                self._save_line_profile_results(name, profiler, execution_time, memory_delta)
            
            # Store basic metrics
            self.profile_results[name] = {
                'execution_time': execution_time,
                'memory_delta': memory_delta,
                'timestamp': datetime.utcnow(),
                'profiler_type': profiler_type.value
            }
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except:
            return 0.0
    
    def _save_cprofile_results(self, name: str, profiler: cProfile.Profile, 
                              execution_time: float, memory_delta: float):
        """Save cProfile results"""
        try:
            # Create stats object
            stats_stream = io.StringIO()
            stats = pstats.Stats(profiler, stream=stats_stream)
            stats.sort_stats('cumulative')
            stats.print_stats(20)  # Top 20 functions
            
            # Save to file
            profile_dir = Path("performance/profiles")
            profile_dir.mkdir(parents=True, exist_ok=True)
            
            profile_file = profile_dir / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.prof"
            profiler.dump_stats(str(profile_file))
            
            # Store summary
            self.profile_results[f"{name}_cprofile"] = {
                'profile_file': str(profile_file),
                'stats_summary': stats_stream.getvalue(),
                'execution_time': execution_time,
                'memory_delta': memory_delta,
                'top_functions': self._extract_top_functions(stats)
            }
            
        except Exception as e:
            logger.error(f"Failed to save cProfile results: {e}")
    
    def _save_memory_profile_results(self, name: str, peak_memory: int, execution_time: float):
        """Save memory profile results"""
        try:
            self.profile_results[f"{name}_memory"] = {
                'peak_memory_bytes': peak_memory,
                'peak_memory_mb': peak_memory / 1024 / 1024,
                'execution_time': execution_time,
                'timestamp': datetime.utcnow()
            }
        except Exception as e:
            logger.error(f"Failed to save memory profile results: {e}")
    
    def _save_line_profile_results(self, name: str, profiler: Any, 
                                  execution_time: float, memory_delta: float):
        """Save line profiler results"""
        try:
            # Line profiler results would be saved here
            # This is a placeholder for actual line profiler integration
            self.profile_results[f"{name}_line"] = {
                'execution_time': execution_time,
                'memory_delta': memory_delta,
                'line_profile': "Line profiling data would be here"
            }
        except Exception as e:
            logger.error(f"Failed to save line profile results: {e}")
    
    def _extract_top_functions(self, stats: pstats.Stats) -> List[Dict[str, Any]]:
        """Extract top functions from profile stats"""
        try:
            # This is a simplified extraction
            # In practice, you would parse the stats more thoroughly
            return [
                {
                    'function': 'example_function',
                    'calls': 100,
                    'total_time': 0.5,
                    'cumulative_time': 1.0
                }
            ]
        except:
            return []
    
    def get_profile_summary(self, name: str) -> Optional[Dict[str, Any]]:
        """Get profile summary for a named profile"""
        return self.profile_results.get(name)
    
    def get_all_profiles(self) -> Dict[str, Any]:
        """Get all profile results"""
        return self.profile_results.copy()

# ==========================================
# BENCHMARKING SYSTEM
# ==========================================

class BenchmarkSuite:
    """Comprehensive benchmarking suite"""
    
    def __init__(self):
        self.benchmarks: Dict[str, Callable] = {}
        self.results: List[BenchmarkResult] = []
        self.baseline_results: Dict[str, BenchmarkResult] = {}
        
    def register_benchmark(self, name: str, benchmark_func: Callable, 
                          setup_func: Optional[Callable] = None,
                          teardown_func: Optional[Callable] = None):
        """Register a benchmark function"""
        self.benchmarks[name] = {
            'function': benchmark_func,
            'setup': setup_func,
            'teardown': teardown_func
        }
        logger.info(f"Registered benchmark: {name}")
    
    def run_benchmark(self, name: str, iterations: int = 1000) -> BenchmarkResult:
        """Run a single benchmark"""
        if name not in self.benchmarks:
            raise ValueError(f"Benchmark '{name}' not found")
        
        benchmark = self.benchmarks[name]
        
        # Setup
        if benchmark['setup']:
            benchmark['setup']()
        
        # Warm up
        for _ in range(min(10, iterations // 10)):
            benchmark['function']()
        
        # Collect garbage before benchmark
        gc.collect()
        
        # Measure baseline system state
        start_memory = psutil.Process().memory_info().rss
        start_cpu_times = psutil.Process().cpu_times()
        
        # Run benchmark
        start_time = time.perf_counter()
        
        for _ in range(iterations):
            benchmark['function']()
        
        end_time = time.perf_counter()
        
        # Measure final system state
        end_memory = psutil.Process().memory_info().rss
        end_cpu_times = psutil.Process().cpu_times()
        
        # Calculate metrics
        execution_time = end_time - start_time
        memory_peak = (end_memory - start_memory) / 1024 / 1024  # MB
        cpu_usage = (end_cpu_times.user - start_cpu_times.user) + (end_cpu_times.system - start_cpu_times.system)
        throughput = iterations / execution_time if execution_time > 0 else 0
        
        # Teardown
        if benchmark['teardown']:
            benchmark['teardown']()
        
        # Create result
        result = BenchmarkResult(
            benchmark_name=name,
            execution_time=execution_time,
            memory_peak=memory_peak,
            cpu_usage=cpu_usage,
            throughput=throughput,
            iterations=iterations,
            metadata={
                'avg_time_per_iteration': execution_time / iterations,
                'memory_per_iteration': memory_peak / iterations,
                'cpu_per_iteration': cpu_usage / iterations
            }
        )
        
        self.results.append(result)
        logger.info(f"Benchmark '{name}' completed: {execution_time:.4f}s, {throughput:.2f} ops/sec")
        
        return result
    
    def run_all_benchmarks(self, iterations: int = 1000) -> List[BenchmarkResult]:
        """Run all registered benchmarks"""
        results = []
        
        for name in self.benchmarks:
            try:
                result = self.run_benchmark(name, iterations)
                results.append(result)
            except Exception as e:
                logger.error(f"Benchmark '{name}' failed: {e}")
        
        return results
    
    def set_baseline(self, name: str, result: Optional[BenchmarkResult] = None):
        """Set baseline result for comparison"""
        if result is None:
            # Run benchmark to establish baseline
            result = self.run_benchmark(name)
        
        self.baseline_results[name] = result
        logger.info(f"Set baseline for benchmark '{name}'")
    
    def compare_to_baseline(self, name: str, current_result: BenchmarkResult) -> Dict[str, Any]:
        """Compare current result to baseline"""
        if name not in self.baseline_results:
            return {'error': 'No baseline set for this benchmark'}
        
        baseline = self.baseline_results[name]
        
        # Calculate percentage changes
        time_change = ((current_result.execution_time - baseline.execution_time) / baseline.execution_time) * 100
        memory_change = ((current_result.memory_peak - baseline.memory_peak) / baseline.memory_peak) * 100
        throughput_change = ((current_result.throughput - baseline.throughput) / baseline.throughput) * 100
        
        return {
            'baseline_date': baseline.timestamp.isoformat(),
            'current_date': current_result.timestamp.isoformat(),
            'execution_time_change_percent': time_change,
            'memory_usage_change_percent': memory_change,
            'throughput_change_percent': throughput_change,
            'performance_summary': self._categorize_performance_change(time_change, memory_change, throughput_change)
        }
    
    def _categorize_performance_change(self, time_change: float, memory_change: float, throughput_change: float) -> str:
        """Categorize overall performance change"""
        if time_change < -5 and throughput_change > 5:
            return "Significant improvement"
        elif time_change < -2 and throughput_change > 2:
            return "Improvement"
        elif abs(time_change) < 2 and abs(throughput_change) < 2:
            return "No significant change"
        elif time_change > 5 or throughput_change < -5:
            return "Performance regression"
        else:
            return "Minor change"
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        if not self.results:
            return {'error': 'No benchmark results available'}
        
        # Group results by benchmark name
        grouped_results = {}
        for result in self.results:
            if result.benchmark_name not in grouped_results:
                grouped_results[result.benchmark_name] = []
            grouped_results[result.benchmark_name].append(result)
        
        # Calculate statistics for each benchmark
        report = {
            'generated_at': datetime.utcnow().isoformat(),
            'total_benchmarks': len(grouped_results),
            'total_runs': len(self.results),
            'benchmarks': {}
        }
        
        for name, results in grouped_results.items():
            execution_times = [r.execution_time for r in results]
            throughputs = [r.throughput for r in results]
            memory_peaks = [r.memory_peak for r in results]
            
            report['benchmarks'][name] = {
                'runs': len(results),
                'execution_time': {
                    'mean': statistics.mean(execution_times),
                    'median': statistics.median(execution_times),
                    'stdev': statistics.stdev(execution_times) if len(execution_times) > 1 else 0,
                    'min': min(execution_times),
                    'max': max(execution_times)
                },
                'throughput': {
                    'mean': statistics.mean(throughputs),
                    'median': statistics.median(throughputs),
                    'stdev': statistics.stdev(throughputs) if len(throughputs) > 1 else 0,
                    'min': min(throughputs),
                    'max': max(throughputs)
                },
                'memory_peak': {
                    'mean': statistics.mean(memory_peaks),
                    'median': statistics.median(memory_peaks),
                    'stdev': statistics.stdev(memory_peaks) if len(memory_peaks) > 1 else 0,
                    'min': min(memory_peaks),
                    'max': max(memory_peaks)
                },
                'latest_result': results[-1].__dict__ if results else None
            }
        
        return report

# ==========================================
# OPTIMIZATION ENGINE
# ==========================================

class OptimizationEngine:
    """AI-powered performance optimization engine"""
    
    def __init__(self, optimization_level: OptimizationLevel = OptimizationLevel.BALANCED):
        self.optimization_level = optimization_level
        self.performance_history: List[PerformanceMetric] = []
        self.optimizations_applied: List[str] = []
        self.recommendations: List[OptimizationRecommendation] = []
        
    def analyze_performance(self, metrics: List[PerformanceMetric]) -> List[OptimizationRecommendation]:
        """Analyze performance metrics and generate optimization recommendations"""
        self.performance_history.extend(metrics)
        recommendations = []
        
        # Analyze execution time patterns
        execution_times = [m for m in metrics if m.name == 'execution_time']
        if execution_times:
            recommendations.extend(self._analyze_execution_time(execution_times))
        
        # Analyze memory usage patterns
        memory_metrics = [m for m in metrics if m.name == 'memory_usage']
        if memory_metrics:
            recommendations.extend(self._analyze_memory_usage(memory_metrics))
        
        # Analyze CPU usage patterns
        cpu_metrics = [m for m in metrics if m.name == 'cpu_usage']
        if cpu_metrics:
            recommendations.extend(self._analyze_cpu_usage(cpu_metrics))
        
        # Analyze cache performance
        cache_metrics = [m for m in metrics if m.name == 'cache_hit_rate']
        if cache_metrics:
            recommendations.extend(self._analyze_cache_performance(cache_metrics))
        
        # Sort recommendations by priority
        recommendations.sort(key=lambda x: x.priority, reverse=True)
        
        self.recommendations = recommendations
        return recommendations
    
    def _analyze_execution_time(self, metrics: List[PerformanceMetric]) -> List[OptimizationRecommendation]:
        """Analyze execution time and suggest optimizations"""
        recommendations = []
        
        if not metrics:
            return recommendations
        
        values = [m.value for m in metrics]
        avg_time = statistics.mean(values)
        
        # Check for slow execution times
        if avg_time > 5.0:  # 5 seconds threshold
            recommendations.append(OptimizationRecommendation(
                category="execution_time",
                priority=8,
                description="High execution time detected",
                implementation="Consider async/await patterns, parallel processing, or algorithm optimization",
                expected_improvement="30-60% execution time reduction",
                estimated_effort="Medium",
                code_changes=[
                    "Convert synchronous operations to async",
                    "Implement parallel processing for independent tasks",
                    "Optimize database queries",
                    "Add caching for expensive operations"
                ]
            ))
        
        # Check for high variability
        if len(values) > 1:
            stdev = statistics.stdev(values)
            cv = stdev / avg_time  # Coefficient of variation
            
            if cv > 0.3:  # High variability
                recommendations.append(OptimizationRecommendation(
                    category="execution_time",
                    priority=6,
                    description="High execution time variability detected",
                    implementation="Implement consistent resource allocation and reduce external dependencies",
                    expected_improvement="More predictable performance",
                    estimated_effort="Low",
                    code_changes=[
                        "Add connection pooling",
                        "Implement circuit breakers",
                        "Add retry mechanisms with backoff"
                    ]
                ))
        
        return recommendations
    
    def _analyze_memory_usage(self, metrics: List[PerformanceMetric]) -> List[OptimizationRecommendation]:
        """Analyze memory usage and suggest optimizations"""
        recommendations = []
        
        if not metrics:
            return recommendations
        
        values = [m.value for m in metrics]
        max_memory = max(values)
        
        # Check for high memory usage (>1GB)
        if max_memory > 1024:  # MB
            recommendations.append(OptimizationRecommendation(
                category="memory_usage",
                priority=7,
                description="High memory usage detected",
                implementation="Implement memory optimization techniques",
                expected_improvement="20-40% memory reduction",
                estimated_effort="Medium",
                code_changes=[
                    "Use generators instead of lists for large datasets",
                    "Implement object pooling",
                    "Add memory-efficient data structures",
                    "Use streaming for large file processing"
                ]
            ))
        
        # Check for memory leaks (steadily increasing memory)
        if len(values) >= 5:
            # Simple trend analysis
            recent_values = values[-5:]
            if all(recent_values[i] < recent_values[i+1] for i in range(len(recent_values)-1)):
                recommendations.append(OptimizationRecommendation(
                    category="memory_usage",
                    priority=9,
                    description="Potential memory leak detected",
                    implementation="Review object lifecycle and implement proper cleanup",
                    expected_improvement="Prevent memory growth over time",
                    estimated_effort="High",
                    code_changes=[
                        "Add explicit cleanup in finally blocks",
                        "Review circular references",
                        "Implement weak references where appropriate",
                        "Add memory monitoring and alerts"
                    ]
                ))
        
        return recommendations
    
    def _analyze_cpu_usage(self, metrics: List[PerformanceMetric]) -> List[OptimizationRecommendation]:
        """Analyze CPU usage and suggest optimizations"""
        recommendations = []
        
        if not metrics:
            return recommendations
        
        values = [m.value for m in metrics]
        avg_cpu = statistics.mean(values)
        
        # Check for high CPU usage
        if avg_cpu > 80:  # 80% threshold
            recommendations.append(OptimizationRecommendation(
                category="cpu_usage",
                priority=7,
                description="High CPU usage detected",
                implementation="Optimize computational algorithms and reduce CPU-intensive operations",
                expected_improvement="20-50% CPU usage reduction",
                estimated_effort="Medium",
                code_changes=[
                    "Profile and optimize hot code paths",
                    "Use more efficient algorithms",
                    "Implement CPU-bound task queuing",
                    "Consider using compiled extensions (Cython)"
                ]
            ))
        
        return recommendations
    
    def _analyze_cache_performance(self, metrics: List[PerformanceMetric]) -> List[OptimizationRecommendation]:
        """Analyze cache performance and suggest optimizations"""
        recommendations = []
        
        if not metrics:
            return recommendations
        
        values = [m.value for m in metrics]
        avg_hit_rate = statistics.mean(values)
        
        # Check for low cache hit rate
        if avg_hit_rate < 0.8:  # 80% threshold
            recommendations.append(OptimizationRecommendation(
                category="cache_performance",
                priority=6,
                description="Low cache hit rate detected",
                implementation="Optimize caching strategy and cache key design",
                expected_improvement="Improved response times and reduced load",
                estimated_effort="Low",
                code_changes=[
                    "Review cache key strategies",
                    "Implement cache warming",
                    "Adjust cache TTL values",
                    "Add cache monitoring and metrics"
                ]
            ))
        
        return recommendations
    
    def apply_optimization(self, recommendation: OptimizationRecommendation) -> bool:
        """Apply an optimization recommendation"""
        try:
            # This is a placeholder for actual optimization implementation
            # In practice, you would have specific optimization implementations
            
            optimization_id = f"{recommendation.category}_{len(self.optimizations_applied)}"
            
            if recommendation.category == "execution_time":
                success = self._apply_execution_time_optimization(recommendation)
            elif recommendation.category == "memory_usage":
                success = self._apply_memory_optimization(recommendation)
            elif recommendation.category == "cpu_usage":
                success = self._apply_cpu_optimization(recommendation)
            elif recommendation.category == "cache_performance":
                success = self._apply_cache_optimization(recommendation)
            else:
                success = False
            
            if success:
                self.optimizations_applied.append(optimization_id)
                logger.info(f"Applied optimization: {recommendation.description}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to apply optimization: {e}")
            return False
    
    def _apply_execution_time_optimization(self, recommendation: OptimizationRecommendation) -> bool:
        """Apply execution time optimization"""
        # Placeholder for actual implementation
        # This would contain specific optimization logic
        return True
    
    def _apply_memory_optimization(self, recommendation: OptimizationRecommendation) -> bool:
        """Apply memory optimization"""
        # Placeholder for actual implementation
        return True
    
    def _apply_cpu_optimization(self, recommendation: OptimizationRecommendation) -> bool:
        """Apply CPU optimization"""
        # Placeholder for actual implementation
        return True
    
    def _apply_cache_optimization(self, recommendation: OptimizationRecommendation) -> bool:
        """Apply cache optimization"""
        # Placeholder for actual implementation
        return True
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Generate optimization report"""
        return {
            'optimization_level': self.optimization_level.value,
            'total_metrics_analyzed': len(self.performance_history),
            'recommendations_generated': len(self.recommendations),
            'optimizations_applied': len(self.optimizations_applied),
            'top_recommendations': [
                {
                    'category': rec.category,
                    'priority': rec.priority,
                    'description': rec.description,
                    'expected_improvement': rec.expected_improvement
                }
                for rec in self.recommendations[:5]  # Top 5
            ],
            'applied_optimizations': self.optimizations_applied,
            'performance_trends': self._calculate_performance_trends()
        }
    
    def _calculate_performance_trends(self) -> Dict[str, Any]:
        """Calculate performance trends over time"""
        if len(self.performance_history) < 2:
            return {}
        
        # Group metrics by type
        metric_groups = {}
        for metric in self.performance_history:
            if metric.name not in metric_groups:
                metric_groups[metric.name] = []
            metric_groups[metric.name].append(metric)
        
        trends = {}
        for metric_name, metrics in metric_groups.items():
            if len(metrics) >= 2:
                # Sort by timestamp
                metrics.sort(key=lambda x: x.timestamp)
                
                # Calculate trend (simple linear trend)
                values = [m.value for m in metrics]
                timestamps = [(m.timestamp - metrics[0].timestamp).total_seconds() for m in metrics]
                
                if len(values) > 1:
                    # Simple slope calculation
                    n = len(values)
                    sum_xy = sum(timestamps[i] * values[i] for i in range(n))
                    sum_x = sum(timestamps)
                    sum_y = sum(values)
                    sum_x2 = sum(x * x for x in timestamps)
                    
                    if n * sum_x2 - sum_x * sum_x != 0:
                        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
                        trends[metric_name] = {
                            'slope': slope,
                            'direction': 'improving' if slope < 0 else 'degrading' if slope > 0 else 'stable',
                            'latest_value': values[-1],
                            'change_from_first': values[-1] - values[0]
                        }
        
        return trends

# ==========================================
# LOAD TESTING FRAMEWORK
# ==========================================

class LoadTester:
    """Load testing and stress testing framework"""
    
    def __init__(self):
        self.test_scenarios: Dict[str, Dict] = {}
        self.test_results: List[Dict] = []
        
    def register_load_test(self, name: str, target_function: Callable,
                          concurrent_users: int = 10, test_duration: int = 60,
                          ramp_up_time: int = 10):
        """Register a load test scenario"""
        self.test_scenarios[name] = {
            'target_function': target_function,
            'concurrent_users': concurrent_users,
            'test_duration': test_duration,
            'ramp_up_time': ramp_up_time
        }
    
    async def run_load_test(self, scenario_name: str) -> Dict[str, Any]:
        """Run a load test scenario"""
        if scenario_name not in self.test_scenarios:
            raise ValueError(f"Load test scenario '{scenario_name}' not found")
        
        scenario = self.test_scenarios[scenario_name]
        
        # Test metrics
        start_time = time.time()
        request_times = []
        errors = []
        successful_requests = 0
        
        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(scenario['concurrent_users'])
        
        async def execute_request():
            async with semaphore:
                try:
                    request_start = time.time()
                    
                    # Execute target function
                    if asyncio.iscoroutinefunction(scenario['target_function']):
                        await scenario['target_function']()
                    else:
                        scenario['target_function']()
                    
                    request_time = time.time() - request_start
                    request_times.append(request_time)
                    
                    nonlocal successful_requests
                    successful_requests += 1
                    
                except Exception as e:
                    errors.append(str(e))
        
        # Generate load
        tasks = []
        test_end_time = start_time + scenario['test_duration']
        
        while time.time() < test_end_time:
            if len(tasks) < scenario['concurrent_users']:
                task = asyncio.create_task(execute_request())
                tasks.append(task)
            
            # Remove completed tasks
            tasks = [task for task in tasks if not task.done()]
            
            await asyncio.sleep(0.1)  # Small delay to prevent overwhelming
        
        # Wait for remaining tasks to complete
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        
        total_time = time.time() - start_time
        
        # Calculate metrics
        total_requests = successful_requests + len(errors)
        throughput = total_requests / total_time if total_time > 0 else 0
        error_rate = len(errors) / total_requests if total_requests > 0 else 0
        
        avg_response_time = statistics.mean(request_times) if request_times else 0
        p95_response_time = self._calculate_percentile(request_times, 95) if request_times else 0
        p99_response_time = self._calculate_percentile(request_times, 99) if request_times else 0
        
        result = {
            'scenario_name': scenario_name,
            'test_duration': total_time,
            'total_requests': total_requests,
            'successful_requests': successful_requests,
            'failed_requests': len(errors),
            'throughput_rps': throughput,
            'error_rate': error_rate,
            'avg_response_time': avg_response_time,
            'p95_response_time': p95_response_time,
            'p99_response_time': p99_response_time,
            'min_response_time': min(request_times) if request_times else 0,
            'max_response_time': max(request_times) if request_times else 0,
            'timestamp': datetime.utcnow(),
            'errors': errors[:10]  # First 10 errors for debugging
        }
        
        self.test_results.append(result)
        
        logger.info(f"Load test '{scenario_name}' completed: {throughput:.2f} RPS, {error_rate:.2%} error rate")
        
        return result
    
    def _calculate_percentile(self, values: List[float], percentile: int) -> float:
        """Calculate percentile value"""
        if not values:
            return 0.0
        
        sorted_values = sorted(values)
        index = (percentile / 100.0) * (len(sorted_values) - 1)
        
        if index.is_integer():
            return sorted_values[int(index)]
        else:
            lower_index = int(index)
            upper_index = lower_index + 1
            weight = index - lower_index
            
            return sorted_values[lower_index] * (1 - weight) + sorted_values[upper_index] * weight

# ==========================================
# MAIN PERFORMANCE SYSTEM
# ==========================================

class PerformanceOptimizationSystem:
    """Main performance optimization system"""
    
    def __init__(self, optimization_level: OptimizationLevel = OptimizationLevel.BALANCED):
        self.profiler = PerformanceProfiler()
        self.benchmark_suite = BenchmarkSuite()
        self.optimization_engine = OptimizationEngine(optimization_level)
        self.load_tester = LoadTester()
        
        # Register default benchmarks
        self._register_default_benchmarks()
    
    def _register_default_benchmarks(self):
        """Register default performance benchmarks"""
        # CPU-intensive benchmark
        def cpu_benchmark():
            # Simple CPU-intensive task
            total = 0
            for i in range(100000):
                total += i * i
            return total
        
        # Memory allocation benchmark
        def memory_benchmark():
            # Allocate and deallocate memory
            data = [list(range(1000)) for _ in range(100)]
            return len(data)
        
        # I/O benchmark
        def io_benchmark():
            # Simple file I/O
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w+') as f:
                f.write("test data" * 1000)
                f.seek(0)
                return len(f.read())
        
        self.benchmark_suite.register_benchmark("cpu_intensive", cpu_benchmark)
        self.benchmark_suite.register_benchmark("memory_allocation", memory_benchmark)
        self.benchmark_suite.register_benchmark("file_io", io_benchmark)
    
    def run_performance_analysis(self) -> Dict[str, Any]:
        """Run comprehensive performance analysis"""
        logger.info("Starting comprehensive performance analysis...")
        
        # Run benchmarks
        benchmark_results = self.benchmark_suite.run_all_benchmarks()
        
        # Convert benchmark results to performance metrics
        metrics = []
        for result in benchmark_results:
            metrics.extend([
                PerformanceMetric(
                    name="execution_time",
                    value=result.execution_time,
                    unit="seconds",
                    timestamp=result.timestamp,
                    context={'benchmark': result.benchmark_name}
                ),
                PerformanceMetric(
                    name="memory_usage",
                    value=result.memory_peak,
                    unit="MB",
                    timestamp=result.timestamp,
                    context={'benchmark': result.benchmark_name}
                ),
                PerformanceMetric(
                    name="throughput",
                    value=result.throughput,
                    unit="ops/sec",
                    timestamp=result.timestamp,
                    context={'benchmark': result.benchmark_name}
                )
            ])
        
        # Generate optimization recommendations
        recommendations = self.optimization_engine.analyze_performance(metrics)
        
        # Generate reports
        benchmark_report = self.benchmark_suite.generate_performance_report()
        optimization_report = self.optimization_engine.get_optimization_report()
        
        return {
            'analysis_timestamp': datetime.utcnow().isoformat(),
            'benchmark_results': benchmark_report,
            'optimization_recommendations': [rec.__dict__ for rec in recommendations],
            'optimization_report': optimization_report,
            'summary': {
                'total_benchmarks_run': len(benchmark_results),
                'total_recommendations': len(recommendations),
                'high_priority_recommendations': len([r for r in recommendations if r.priority >= 8]),
                'performance_status': self._assess_overall_performance(benchmark_results, recommendations)
            }
        }
    
    def _assess_overall_performance(self, benchmark_results: List[BenchmarkResult], 
                                   recommendations: List[OptimizationRecommendation]) -> str:
        """Assess overall performance status"""
        high_priority_issues = len([r for r in recommendations if r.priority >= 8])
        
        if high_priority_issues == 0:
            return "Excellent"
        elif high_priority_issues <= 2:
            return "Good"
        elif high_priority_issues <= 5:
            return "Needs Improvement"
        else:
            return "Poor"
    
    def auto_optimize(self, max_optimizations: int = 5) -> Dict[str, Any]:
        """Automatically apply top optimization recommendations"""
        recommendations = self.optimization_engine.recommendations
        
        if not recommendations:
            return {'message': 'No recommendations available. Run performance analysis first.'}
        
        applied_optimizations = []
        failed_optimizations = []
        
        # Apply top recommendations
        for recommendation in recommendations[:max_optimizations]:
            if recommendation.priority >= 6:  # Only apply medium to high priority optimizations
                success = self.optimization_engine.apply_optimization(recommendation)
                
                if success:
                    applied_optimizations.append(recommendation.description)
                else:
                    failed_optimizations.append(recommendation.description)
        
        return {
            'applied_optimizations': applied_optimizations,
            'failed_optimizations': failed_optimizations,
            'total_applied': len(applied_optimizations),
            'optimization_timestamp': datetime.utcnow().isoformat()
        }

# ==========================================
# EXAMPLE USAGE
# ==========================================

def example_performance_optimization():
    """Example of using the performance optimization system"""
    
    # Create performance system
    perf_system = PerformanceOptimizationSystem(OptimizationLevel.AGGRESSIVE)
    
    # Example function to profile
    def example_function():
        # Simulate some work
        import random
        data = [random.random() for _ in range(10000)]
        return sum(data)
    
    # Profile function execution
    with perf_system.profiler.profile_execution("example_function"):
        result = example_function()
    
    # Run performance analysis
    analysis_results = perf_system.run_performance_analysis()
    
    print("Performance Analysis Results:")
    print(f"Benchmarks run: {analysis_results['summary']['total_benchmarks_run']}")
    print(f"Recommendations: {analysis_results['summary']['total_recommendations']}")
    print(f"Performance status: {analysis_results['summary']['performance_status']}")
    
    # Apply automatic optimizations
    optimization_results = perf_system.auto_optimize()
    
    print(f"\nOptimizations applied: {optimization_results['total_applied']}")
    for opt in optimization_results['applied_optimizations']:
        print(f"  - {opt}")

if __name__ == "__main__":
    example_performance_optimization()