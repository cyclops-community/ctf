### Documentation build instructions
Install CTF python library system-wide or add appropriate paths, then run.
```sh
pip install -r requirements.txt
sphinx-apidoc -f -o . ../ctf && make html
```
