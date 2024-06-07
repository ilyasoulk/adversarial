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
    # If the code is empty, return a specific score
    if code == '':
        print("Code snippet is empty, parsing failed")
        return -2

    # Check for syntax errors
    try:
        compile(code, '<string>', 'exec')
    except SyntaxError as e:
        print(f"Syntax Error: {e}")
        return -1.0

    # Combine the code and unit tests into a single script
    script = code + '\n' + unit_tests
    namespace = {}

    # Check for NameError and other runtime errors
    try:
        exec(script, namespace)
    except NameError as e:
        print(f"Name Error: {e}")
        return -0.6
    except RuntimeError as e:
        print(f"Runtime Error: {e}")
        return -0.6
    except AssertionError as e:
        print(f"Assertion Error: {e}")
        return -0.3
    except Exception as e:
        print(f"Other Error: {e}")
        return -0.4

    # If no exceptions, return a positive score
    print("Code snippet passed all unit tests")
    return 1.0