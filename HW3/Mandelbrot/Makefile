CC=nvcc                                                                      
TARGET=mandelbrot              
SRC=mandelbrot.cu                    



all: $(TARGET)


$(TARGET): $(SRC)
	$(CC) -o $@ $^


clean:
	rm -f $(TARGET)