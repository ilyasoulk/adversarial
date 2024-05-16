def reward_model(code, unit_tests):
    '''
    This function should take in a code snippet and a list of unit tests.
    It should return a score that represents how well the code snippet.
    The score is yet to be defined.
    I assume the score will depend on 2 things:
    1. Are there any syntax errors in the code snippet ?
    2. Does the code snippet pass all the unit tests ?
    The score should take into account both of these factors.
    '''
    # TODO: Implement reward model
    try:
        compile(code, '<string>', 'exec')
    except SyntaxError as e:
        return -1.0, f"Syntax Error: {e}"

    script = code + '\n' + unit_tests
    try:
        namespace = {}
        exec(script, namespace)
    except RuntimeError as e:
        return -0.6, f"Runtime Error: {e}"

    try:
        exec(script, namespace)
    except AssertionError as e:
        return -0.3, f"Assertion Error: {e}"
    
    return 1.0, "Code snippet passed all unit tests"