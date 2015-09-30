#
# Makefile for Billing Agent
#
# $Id: Makefile 125 2009-03-01 16:37:11Z cdsheen $

OSNAME=centos5

#DEFINES=-D_PRODUCTION


TARGET=rb

CC=g++

RM=rm

#revision := $(shell svnversion|sed 's/[\:MS].*//')

#CFLAGS=-Wall -g -fPIC -I $(DEFINES) -DSVN_REVISION=$(revision) -D_LARGEFILE_SOURCE -D_FILE_OFFSET_BITS=64

#CFLAGS=-Wall -g -fPIC -I $(DEFINES) -D_LARGEFILE_SOURCE -D_FILE_OFFSET_BITS=64
CFLAGS=-Wall -g  -std=gnu++11
#LDFLAGS=-lpthread -shared -fPIC
LDFLAGS=-llept -ltesseract `pkg-config --static --cflags --libs opencv`
OBJS=ConnectedComponent.o RobustTextDetection.o main.o


.SUFFIXES:	.cpp

.c.o:	
	$(CC) $(CFLAGS) -c $<

.cpp.o:	
	$(CC) $(CFLAGS) -c $<

all: $(TARGET)

rb:	$(OBJS)
	$(CC) $(OBJS) $(LDFLAGS) -o $(TARGET)


clean:
	$(RM) *.o *~ $(TARGET)
