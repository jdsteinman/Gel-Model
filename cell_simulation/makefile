run:
	#nohup mpirun -np 16 python3 parallel.py >> log &
	#nohup mpirun -np 8 python3 three_field.py >> log &
	nohup mpirun -np 16 python3 single_field.py >> log &
clean:
	-rm log
