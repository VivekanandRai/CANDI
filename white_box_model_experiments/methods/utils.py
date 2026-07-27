import json
import logging
import requests
import ast
import time
import os

logger = logging.getLogger(__name__)


def get_response(input_prompt, model, seed=42, max_tokens=500, temperature=1.0, response_format=None, max_retries=3, retry_delay=5):
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {os.environ.get('OPENROUTER_API_KEY')}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [
                {"role": "user", "content": input_prompt},
                ],
                "response_format": response_format,
                "max_tokens": max_tokens,
                "temperature": temperature,
                    
            },
            timeout=(30, 120),  # (connection timeout, read timeout) in seconds
            )
            
            # Check for HTTP errors - retry on 5xx errors
            if response.status_code >= 500:
                if attempt < max_retries - 1:
                    print(f"API returned {response.status_code}, retrying in {retry_delay}s... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(retry_delay)
                    continue
                raise RuntimeError(f"OpenRouter API error: status={response.status_code}, response={response.text}")
            
            if response.status_code != 200:
                raise RuntimeError(f"OpenRouter API error: status={response.status_code}, response={response.text}")
            
            # Check for empty response
            if not response.text or response.text.strip() == "":
                raise RuntimeError("OpenRouter API returned empty response")
            
            try:
                data = response.json()
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Failed to parse OpenRouter API response as JSON: {response.text[:500]}") from e
            
            # Check for API-level errors
            if "error" in data:
                raise RuntimeError(f"OpenRouter API error: {data['error']}")
            
            if "choices" not in data or len(data["choices"]) == 0:
                raise RuntimeError(f"OpenRouter API returned no choices: {data}")
            
            return data["choices"][0]["message"]["content"]
        
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                print(f"Request failed: {e}, retrying in {retry_delay}s... (attempt {attempt + 1}/{max_retries})")
                time.sleep(retry_delay)
                continue
            raise RuntimeError(f"OpenRouter API request failed after {max_retries} attempts: {e}")
    
    raise RuntimeError(f"OpenRouter API failed after {max_retries} attempts")

def parse_json_response(response_string):
    """
    Robustly parse JSON responses that may contain Python-style (True/False) 
    or JSON-style (true/false) boolean values.
    
    Args:
        response_string: String containing JSON data
        
    Returns:
        Parsed dictionary
    """
    # Strip whitespace and common wrappers
    response_string = response_string.strip()
    
    # Remove markdown code blocks if present
    if response_string.startswith('```'):
        lines = response_string.split('\n')
        # Remove first line if it's ```json or ```
        if lines[0].startswith('```'):
            lines = lines[1:]
        # Remove last line if it's ```
        if lines and lines[-1].strip() == '```':
            lines = lines[:-1]
        response_string = '\n'.join(lines).strip()
    
    try:
        # First try standard JSON parsing
        return json.loads(response_string)
    except json.JSONDecodeError as e:
        # If that fails, replace Python booleans with JSON booleans and try again
        try:
            normalized = response_string.replace('True', 'true').replace('False', 'false')
            return json.loads(normalized)
        except json.JSONDecodeError:
            # If still failing, try ast.literal_eval as last resort
            try:
                return ast.literal_eval(response_string)
            except (ValueError, SyntaxError) as ast_error:
                # Re-raise the original JSON error with more context
                raise json.JSONDecodeError(
                    f"Failed to parse as JSON or Python literal. Original error: {e}. AST error: {ast_error}",
                    response_string,
                    e.pos
                )

def hallucination_check(context, input_query, target_response, hallucination_evaluator_model, correct_answer=None, return_justification=False):
    '''This function checks whether the target response has any hallucination behavior.
    The prompt is inspired by 'Siren's Song in the AI Ocean: A Survey on Hallucination in Large Language Models' (https://arxiv.org/abs/2309.01219) and
    'A Survey on Hallucination in Large Language Models: Principles, Taxonomy, Challenges, and Open Questions' (https://arxiv.org/abs/2311.05232).'''

    input_prompt = f"""
        You are an expert evaluator for hallucination detection in Retrieval-Augmented Generation (RAG) systems.

        You will be given:
        1. A user query
        2. Retrieved context
        3. A generated answer
        4. (Optionally) the correct answer

        Your task is to evaluate ONLY the generated answer's faithfulness to the retrieved context.

        Context: {context}
        User: {input_query}
        Response to evaluate: "{target_response}"
        Correct answer: "{correct_answer if correct_answer is not None else 'Not Available'}"

        
        Important scope:
        - Evaluate only the generation step, NOT retrieval quality.
        - Do NOT judge whether the retriever should have found better documents.
        - Do NOT use outside knowledge.
        - Treat the retrieved context as the only allowed evidence source.
        - If the context is incomplete, vague, or missing key information, do NOT penalize the answer for missing facts unless it invents or overstates them.
        - A good answer may say that the context does not contain enough information.

        Your goal:
        Determine whether the generated answer stays grounded in the retrieved context and avoids hallucination.

        Definitions:
        - "Material claim" means any substantive factual assertion, explanation, recommendation, comparison, numerical statement, quoted statement, causal statement, or conclusion that affects the meaning of the answer.
        - Ignore harmless stylistic phrasing unless it introduces a material unsupported claim.
        - Evaluate only what is actually said in the generated answer.

        Scoring rubric:
        You must assign a boolean score (true or false) for each of the following five criteria.

        1. contextual_support
        Question: Are all material claims in the generated answer supported by the retrieved context?
        - Score true if all material claims are supported, directly stated, or clearly entailed by the retrieved context. It is okay if the claims made are easily or directly implied by the context, or are common knowledge that we can reasonably expect an average person to understand.
        - Score false if any material claim is unsupported, invented, or not justified by the retrieved context.

        2. no_fabricated_specifics
        Question: Does the generated answer avoid inventing unsupported specific details?
        Examples of specifics include:
        - numbers
        - dates
        - names
        - locations
        - percentages
        - study findings
        - quotations
        - citations
        - procedural details
        - technical attributes
        - timelines
        - policies
        - product features
        - Score true if no unsupported specific details are introduced.
        - Score false if any unsupported specific detail is introduced.

        3. no_contradiction
        Question: Does the generated answer avoid contradicting the retrieved context?
        - Score true if the answer does not conflict with the retrieved context.
        - Score false if any part of the answer contradicts the retrieved context.

        4. proper_uncertainty
        Question: When the retrieved context is insufficient to fully answer the query, does the generated answer appropriately express uncertainty or limit its claims?
        - Score true if the answer remains appropriately bounded, acknowledges missing information when needed, and does not fill gaps with confident unsupported claims.
        - Score false if the answer overclaims, presents uncertain information as certain, or confidently fills gaps not supported by the context.

        5. faithful_synthesis
        Question: If the generated answer summarizes, combines, or infers across multiple parts of the retrieved context, is that synthesis faithful?
        - Score true if the summary/inference is justified by the retrieved context and does not distort, overgeneralize, or overextend the evidence.
        - Score false if the answer makes unjustified inferences, misleading summaries, exaggerated conclusions, or unsupported generalizations.

        Evaluation instructions:
        - Read the user query, retrieved context, and generated answer carefully.
        - Focus only on whether the answer is grounded in the retrieved context.
        - Do not reward correctness from outside knowledge.
        - Do not punish the answer simply because the context is limited.
        - Penalize unsupported certainty, invented details, contradiction, and unjustified inference.
        - Be conservative: if support is unclear, do not assume it exists.
        - A single material hallucination is enough to score 0 on the relevant criterion.
        - Base your rationale on the retrieved context only.

        Scoring procedure:
        - Assign each criterion either true or false.

        Return strict JSON:
        {{
          "contextual_support": true or false,
          "contextual_support_justification": "brief explanation for this criterion",
          "no_fabricated_specifics": true or false,
          "no_fabricated_specifics_justification": "brief explanation for this criterion",
          "no_contradiction": true or false,
          "no_contradiction_justification": "brief explanation for this criterion",
          "proper_uncertainty": true or false,
          "proper_uncertainty_justification": "brief explanation for this criterion",
          "faithful_synthesis": true or false,
          "faithful_synthesis_justification": "brief explanation for this criterion"
        }}
        """

    criteria_keys = (
            "contextual_support",
            "no_fabricated_specifics",
            "no_contradiction",
            "proper_uncertainty",
            "faithful_synthesis",
        )
    hallucination_response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "hallucination_score",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "contextual_support": {"type": "boolean"},
                        "contextual_support_justification": {"type": "string"},
                        "no_fabricated_specifics": {"type": "boolean"},
                        "no_fabricated_specifics_justification": {"type": "string"},
                        "no_contradiction": {"type": "boolean"},
                        "no_contradiction_justification": {"type": "string"},
                        "proper_uncertainty": {"type": "boolean"},
                        "proper_uncertainty_justification": {"type": "string"},
                        "faithful_synthesis": {"type": "boolean"},
                        "faithful_synthesis_justification": {"type": "string"},
                    },
                    "required": [
                        *criteria_keys,
                        "contextual_support_justification",
                        "no_fabricated_specifics_justification",
                        "no_contradiction_justification",
                        "proper_uncertainty_justification",
                        "faithful_synthesis_justification",
                    ],
                    "additionalProperties": False,
                },
            },
        }
    
    justification = None
    hallucination_detected = False
    criteria = {}

    try:
        response = get_response(
            input_prompt=input_prompt,
            model=hallucination_evaluator_model,
            # The rubric asks for five booleans plus five free-text justifications.
            # Too small a budget truncates the reply, which then fails to parse and
            # would be silently scored as "no hallucination".
            max_tokens=1200,
            temperature=0.0,
            response_format=hallucination_response_format
        )

        # Convert string to dictionary - handles both JSON and Python boolean formats
        parsed_result = parse_json_response(response)

        # Each criterion is True when the response is faithful on that axis, so a
        # single False anywhere means the response hallucinated.
        criteria = {
            criterion: bool(parsed_result.get(criterion, True))
            for criterion in criteria_keys
        }
        failed = [criterion for criterion, passed in criteria.items() if not passed]
        hallucination_detected = bool(failed)

        # The rubric emits one justification per criterion. Report the ones that
        # explain the verdict: the failures, or the synthesis rationale if clean.
        reported = failed if failed else ["faithful_synthesis"]
        parts = []
        for criterion in reported:
            text = parsed_result.get(f"{criterion}_justification")
            if isinstance(text, str) and text.strip():
                parts.append(f"[{criterion}] {text.strip()}" if failed else text.strip())
        justification = " ".join(parts) if parts else None

    except Exception:
        # Do not mask judge failures as a clean verdict - surface them so a run
        # with a broken judge is visibly broken rather than silently scoring 0.
        logger.warning(
            "Hallucination judge (%s) failed; recording no hallucination for this call.",
            hallucination_evaluator_model,
            exc_info=True,
        )

    if return_justification:
        return {
            "hallucination_detected": hallucination_detected,
            "justification": justification,
            "criteria": criteria,
        }
    else:
        return hallucination_detected