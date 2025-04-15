TEACHER_PROMPT = """
dio mus
"""

def get_teacher_prompt(target_user_name: str, user_context: str) -> str:
    return TEACHER_PROMPT.format(target_user_name=target_user_name, user_context=user_context)