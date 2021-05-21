#pragma once
#ifndef TIMER_H
#define TIMER_H

#include <sys/time.h>
#include <stdlib.h>

//define the main timer, uses gettimeofday function to compute time
class MainTimer
{
  public:
    inline MainTimer();
    inline ~MainTimer();

    /** Starts the timer */
    inline void start();

    /** Stops the timer */
    inline void stop();

    /** Checks to see if the timer is stopped */
    inline bool isStopped() const;

    /** Returns the elapsed time since the timer was started, or the time interval
      between calls to start() and stop().
      */
    inline double getElapsedMicroseconds() const;
    inline double getElapsedMilliseconds() const;
    inline double getElapsedSeconds() const;

    /** Decrease the timer by one second (useful for FPS computations) */
    inline void decreaseBy1Second();

    /** Decrease the timer by a custom time (in seconds) */
    inline void decreaseBy(double seconds);

  private:
    bool isStopped_;
    timeval start_;
    timeval end_;
};

MainTimer::MainTimer() 
{
  start_.tv_sec = 0;
  start_.tv_usec = 0;
  isStopped_ = true;
}

MainTimer::~MainTimer() { }

void MainTimer::start() 
{
  gettimeofday(&start_, NULL); // Get the starting time
  isStopped_ = false;
}

void MainTimer::stop() 
{
  gettimeofday(&end_, NULL); // Get the ending time
  isStopped_ = true;
}

bool MainTimer::isStopped() const 
{
  return isStopped_;
}

double MainTimer::getElapsedMicroseconds() const
{
  timeval time;
  double microSecond = 0;

  if(!isStopped_) 
  {
    gettimeofday(&time, NULL);
  }
  else
    time = end_;
  microSecond = (time.tv_sec * 1000000.0 + time.tv_usec) - (start_.tv_sec * 1000000.0 + start_.tv_usec);
  return microSecond;
}

double MainTimer::getElapsedMilliseconds() const
{
  return getElapsedMicroseconds() / 1000.0;
}

double MainTimer::getElapsedSeconds() const
{
  return getElapsedMicroseconds() / 1000000.0;
}
void MainTimer::decreaseBy1Second()
{
  start_.tv_sec++;
}
/** Decrease the timer by a custom time (in seconds) */
void MainTimer::decreaseBy(double seconds)
{
  start_.tv_sec += int(seconds);
  start_.tv_usec += int(1000000*(seconds-int(seconds)));
}

typedef MainTimer Timer;

#endif
