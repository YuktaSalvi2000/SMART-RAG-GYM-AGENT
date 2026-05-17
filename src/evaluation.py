"""
src/evaluation.py
Response quality evaluation pipeline
Measures: relevance, hallucinations, quality, latency
"""

import logging
import time
from typing import Dict, Any, Optional
from src.llm_setup import get_model_adaptive

logger = logging.getLogger(__name__)

# ========================================
# RELEVANCE EVALUATION
# ========================================

def evaluate_answer_relevance(question: str, answer: str, llm=None) -> float:
    """
    Evaluate if answer answers the question (1-10)
    
    Args:
        question: User's question
        answer: LLM's answer
        llm: LLM instance (optional)
    
    Returns:
        Score 1-10
    """
    
    if not llm:
        llm = get_model_adaptive()
    
    try:
        prompt = f"""Rate how well this answer addresses the question on a scale of 1-10.

QUESTION: {question}

ANSWER: {answer}

Rate 1-10 where:
- 1-3: Completely irrelevant
- 4-6: Partially relevant, missing key parts
- 7-8: Mostly relevant, minor gaps
- 9-10: Fully relevant, directly addresses question

Respond ONLY with a number (1-10):"""

        response = llm.invoke(prompt).content.strip()
        score = float(response.split('\n')[0].strip())
        return min(10, max(1, score))
    
    except Exception as e:
        logger.warning(f"Relevance evaluation failed: {e}")
        return 5.0


# ========================================
# HALLUCINATION DETECTION
# ========================================

def detect_hallucinations(answer: str, user_profile: dict, llm=None) -> Dict[str, Any]:
    """
    Detect potential hallucinations in response
    
    Args:
        answer: LLM response
        user_profile: User's profile (for context)
        llm: LLM instance (optional)
    
    Returns:
        Dict with hallucination_count and issues
    """
    
    if not llm:
        llm = get_model_adaptive()
    
    # Rule-based checks first (fast)
    rule_based_hallucinations = _rule_based_hallucination_check(answer, user_profile)
    
    # LLM-based check (thorough)
    try:
        health_issues = user_profile.get('health_issues', 'none') if user_profile else 'none'
        
        prompt = f"""Check if this fitness advice contains false or dangerous claims.

HEALTH ISSUES: {health_issues}

RESPONSE:
{answer}

Check for:
1. Unrealistic claims (e.g., "lose 20kg/week")
2. Dangerous recommendations for health issues
3. False fitness facts
4. Contradictions with safety guidelines

List any hallucinations found. If none, respond "No hallucinations detected"."""

        response = llm.invoke(prompt).content.strip()
        
        llm_hallucinations = []
        if "No hallucinations" not in response:
            llm_hallucinations = [line.strip() for line in response.split('\n') if line.strip()]
        
        all_issues = rule_based_hallucinations + llm_hallucinations
        
        return {
            "hallucination_count": len(all_issues),
            "issues": all_issues,
            "rule_based_count": len(rule_based_hallucinations),
            "llm_based_count": len(llm_hallucinations)
        }
    
    except Exception as e:
        logger.warning(f"LLM-based hallucination check failed: {e}")
        
        # Fall back to rule-based only
        return {
            "hallucination_count": len(rule_based_hallucinations),
            "issues": rule_based_hallucinations,
            "rule_based_count": len(rule_based_hallucinations),
            "llm_based_count": 0
        }


def _rule_based_hallucination_check(answer: str, user_profile: dict) -> list:
    """Rule-based hallucination detection (fast, no LLM call)"""
    
    issues = []
    answer_lower = answer.lower()
    
    # Check 1: Unrealistic weight loss claims
    import re
    weight_losses = re.findall(r'(\d+)\s*kg.*(?:week|month)', answer_lower)
    for w in weight_losses:
        w_num = int(w)
        if w_num > 2:  # More than 2kg per week is unrealistic
            issues.append(f"Unrealistic weight loss claim: {w}kg per period")
    
    # Check 2: Dangerous exercises for injuries
    if user_profile:
        health_issues = (user_profile.get('health_issues', '') or '').lower()
        
        if 'back' in health_issues:
            if 'deadlift' in answer_lower or 'squat' in answer_lower:
                if 'heavy' in answer_lower or 'max' in answer_lower:
                    issues.append("Recommends heavy back exercises despite back pain")
        
        if 'knee' in health_issues:
            if 'jumping' in answer_lower or 'plyometric' in answer_lower:
                issues.append("Recommends high-impact exercises despite knee issues")
    
    # Check 3: Contradictory advice
    if 'should not' in answer_lower and 'should' in answer_lower:
        # Only flag if about same topic (simple heuristic)
        should_count = len(re.findall(r'should\s+(\w+)', answer_lower))
        should_not_count = len(re.findall(r'should\s+not\s+(\w+)', answer_lower))
        if should_count > 0 and should_not_count > 0:
            # Low-confidence check - not added unless very clear contradiction
            pass
    
    return issues


# ========================================
# QUALITY METRICS
# ========================================

def evaluate_answer_quality(
    user_question: str,
    answer: str,
    user_profile: dict,
    llm=None
) -> Dict[str, float]:
    """
    Comprehensive quality evaluation
    
    Args:
        user_question: User's question
        answer: LLM's answer
        user_profile: User's profile
        llm: LLM instance (optional)
    
    Returns:
        Dict with all quality scores
    """
    
    start_time = time.time()
    
    # Evaluate relevance
    relevance_score = evaluate_answer_relevance(user_question, answer, llm)
    
    # Detect hallucinations
    hallucination_result = detect_hallucinations(answer, user_profile, llm)
    hallucination_count = hallucination_result['hallucination_count']
    
    # Calculate quality score
    # High relevance (8+) = good
    # Low hallucinations (0-2) = good
    quality_score = (relevance_score + (10 - min(hallucination_count * 2, 10))) / 2
    
    latency_ms = (time.time() - start_time) * 1000
    
    return {
        "answer_relevance": relevance_score,
        "hallucination_count": hallucination_count,
        "quality_score": quality_score,
        "latency_ms": latency_ms,
        "overall_score": quality_score  # For AWS metrics
    }


# ========================================
# CONTEXTUAL QUALITY CHECK
# ========================================

def check_profile_usage(answer: str, user_profile: dict) -> Dict[str, Any]:
    """
    Check if response properly uses user profile context
    
    Returns dict with:
    - profile_mentioned: bool (did it mention key constraints)
    - fitness_level_mentioned: bool
    - health_issues_mentioned: bool
    - quality_score: 0-10
    """
    
    if not user_profile:
        return {
            "profile_mentioned": False,
            "fitness_level_mentioned": False,
            "health_issues_mentioned": False,
            "quality_score": 0
        }
    
    answer_lower = answer.lower()
    
    # Check if profile constraints mentioned
    health_issues = (user_profile.get('health_issues', '') or '').lower()
    fitness_level = (user_profile.get('fitness_level', '') or '').lower()
    
    health_mentioned = False
    if health_issues and health_issues != 'none':
        # Simple check: does response mention any part of health issues
        for issue in health_issues.split(','):
            issue = issue.strip()
            if issue and issue in answer_lower:
                health_mentioned = True
                break
    
    fitness_mentioned = fitness_level in answer_lower if fitness_level else False
    
    quality = 0
    if health_mentioned:
        quality += 5
    if fitness_mentioned:
        quality += 5
    
    return {
        "profile_mentioned": health_mentioned or fitness_mentioned,
        "fitness_level_mentioned": fitness_mentioned,
        "health_issues_mentioned": health_mentioned,
        "quality_score": quality
    }


# ========================================
# METRICS COLLECTION
# ========================================

def collect_all_metrics(
    user_question: str,
    answer: str,
    user_profile: dict,
    llm=None,
    latency_ms: Optional[float] = None
) -> Dict[str, Any]:
    """
    Collect all evaluation metrics
    
    Returns:
        Dict with all metrics for logging/analysis
    """
    
    start = time.time()
    
    # Quality evaluation
    quality_metrics = evaluate_answer_quality(user_question, answer, user_profile, llm)
    
    # Profile usage check
    profile_check = check_profile_usage(answer, user_profile)
    
    # Overall scoring
    overall_score = (
        quality_metrics['answer_relevance'] * 0.4 +
        quality_metrics['quality_score'] * 0.4 +
        profile_check['quality_score'] * 0.2
    ) / 10 * 10  # Normalize to 0-10
    
    total_eval_latency = (time.time() - start) * 1000
    
    return {
        # Answer quality
        "answer_relevance": quality_metrics['answer_relevance'],
        "quality_score": quality_metrics['quality_score'],
        "latency_ms": latency_ms or quality_metrics['latency_ms'],
        
        # Hallucinations
        "hallucination_count": quality_metrics['hallucination_count'],
        "hallucination_issues": [],  # Could add from hallucination_result
        
        # Profile usage
        "profile_mentioned": profile_check['profile_mentioned'],
        "fitness_level_mentioned": profile_check['fitness_level_mentioned'],
        "health_issues_mentioned": profile_check['health_issues_mentioned'],
        
        # Overall
        "overall_score": min(10, overall_score),
        
        # Metadata
        "evaluation_latency_ms": total_eval_latency,
        "prompt_version": "v1.0"
    }