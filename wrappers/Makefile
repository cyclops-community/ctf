all: CTF.so

CTF.so: ../src ../lib CTF.pyx 
	$(MAKE) ctflibso -C ../
	LDFLAGS="-L../lib" python setup.py build_ext  --inplace

