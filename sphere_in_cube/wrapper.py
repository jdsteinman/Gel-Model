import subprocess

const = [0.5, 0.1, 0.01, 0.001]
for c in const:
    subprocess.run("python3 parallel.py " + str(c), shell=True)