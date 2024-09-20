CC=nvcc                                             
LDFLAGS=-lcurand                          
TARGET=pi                
SRC=pi.cu                    



all: $(TARGET)


$(TARGET): $(SRC)
	$(CC) -o $@ $^ $(LDFLAGS)


clean:
	rm -f $(TARGET)