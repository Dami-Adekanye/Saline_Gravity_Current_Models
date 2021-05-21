#pragma once
#include <stdio.h>
#include <iostream>
using std::cout;
using std::endl;
#include "Timer.hpp"

namespace PrintMode
{
  enum Enum
  {
    NONE,
    SAME_LINE,
    NEW_LINE
  };
}

class Statistics
{
  private:
    //Counts the number of frames per second
    int counter_;
    //Counts the total number of calls to update()
    unsigned long int totalCounter_;
    //Timer to measure how long has passed since the last frame
    Timer  frameTimer_;
    //Timer to measure how long has passed since the last print of informations
    Timer timer_;
    //Measure the total run time of the simulation
    Timer totalTimer_;
    //Time between two prints (in s)
    double print_delay_;
    //Number of nodes in the domain (to compute MLUPS)
    unsigned int nbr_of_nodes_;
    //states if the recording as started
    bool started_;
    //how the informations are printed in the console
    PrintMode::Enum printmode_;
    //something to display before the text
    std::string prefix_;
    //true if the number of lattice update per second needs to be displayed as well
    bool displayLUPS_;
  public:
    //constructor
    Statistics(unsigned int nbr_of_nodes, double print_delay = 1.0)
      : counter_(0), totalCounter_(0),
        nbr_of_nodes_(nbr_of_nodes),
        print_delay_(print_delay),
        started_(false),
        displayLUPS_(false),
        printmode_(PrintMode::SAME_LINE),
        prefix_("")
    {
    }
    //set the output mode
    inline void setOutputMode(PrintMode::Enum mode) {
      printmode_ = mode;
    }
    //enable LUPS display
    inline void enableLUPS() { displayLUPS_ = true; }
    //call this function at each frame return the duration of the frame
    inline double update(std::string informations = "")
    {
      cudaDeviceSynchronize();
      if(!started_) {
        started_ = true;
        frameTimer_.start();
        timer_.start();
        totalTimer_.start();
      }
      //time since the last frame
      double frame_time = frameTimer_.getElapsedSeconds();
      frameTimer_.start();
      counter_++;
      totalCounter_++;
      //cout << counter_ << "; ";
      if(timer_.getElapsedSeconds()>=print_delay_)
      {
        switch(printmode_)
        {
          case PrintMode::NONE:
            break;
          case PrintMode::SAME_LINE:
            cout << "\r"<<prefix_<<"MLUPS: " << int(0.000001*counter_*nbr_of_nodes_/print_delay_) ;
            if(displayLUPS_)
              cout << ", LUPS: " << int(counter_/print_delay_);
            cout << " " << informations << "      ";
            fflush(stdout);
            break;
          case PrintMode::NEW_LINE:
            cout << prefix_<<"MLUPS: " << int(0.000001*counter_*nbr_of_nodes_/print_delay_) ;
            if(displayLUPS_)
              cout << ", LUPS: " << int(counter_/print_delay_);
            cout << " "<< informations << endl;
            break;
        }
        counter_ = 0;
        timer_.decreaseBy(print_delay_);
      }
      return frame_time;
    }
    //set some text to be displayed before the rest
    inline void setPrefix(std::string prefix) {
      prefix_ = prefix;
    }
    //stop the total timer (call at the end of the simulation)
    inline void stopTimer() {
      totalTimer_.stop();
    }
    //compute the average MLUPS
    inline int getAverageMLUPS() {
      return int(0.000001*totalCounter_*nbr_of_nodes_/totalTimer_.getElapsedSeconds());
    }
};
