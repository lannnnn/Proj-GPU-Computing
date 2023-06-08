CC = mpic++
COPTS = -g -std=c++11 -fPIC -O0 -Wall -Wstrict-aliasing -fstrict-aliasing

PATH_INCLUDES = /home/shuxin.zheng/local/include
PATH_LIBS = /home/shuxin.zheng/local/lib -lmetis -lGKlib -lparmetis 

INCLUDES = -I${PATH_INCLUDES} 
LFLAGS =  -L${PATH_LIBS} 

TARGET = test_parMetis
SOURCES = test_parMetis.cpp
OBJECTS = $(SOURCES:.cpp=.o)

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CC) $(COPTS) $(INCLUDES) -o $@ $^ $(LFLAGS)

.cpp.o:
	$(CC) $(COPTS) $(INCLUDES) -c $< -o $@ $(LFLAGS)

clean:
	rm -f $(TARGET) $(OBJECTS)