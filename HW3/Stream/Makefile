CC=nvcc                                                                      
TARGET=stream                
SRC=stream.cu                    



all: $(TARGET)


$(TARGET): $(SRC)
	$(CC) -o $@ $^


clean:
	rm -f $(TARGET)