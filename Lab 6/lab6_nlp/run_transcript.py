import builtins
import runpy
import sys

script = sys.argv[1]
inputs = iter(sys.argv[2].split("|"))
sys.argv = [script] + sys.argv[3:]


def scripted_input(prompt=""):
    try:
        value = next(inputs)
    except StopIteration:
        raise EOFError
    print(prompt + value)
    return value


builtins.input = scripted_input
runpy.run_path(script, run_name="__main__")
