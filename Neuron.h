//
//  Neuron.h
//  Point Detector
//
//  Created by Yik Tung Ho on 1/8/14.
//  Copyright (c) 2015 YTH. All rights reserved.
//

#ifndef Point_Detector_Neuron_h
#define Point_Detector_Neuron_h

#include <array>
#include <vector>
#include <cmath>
#include <exception>
#include <sstream>

typedef std::array<double,2> Input;

// abstract base class of a neural processor taking 2 inputs and 1 output
class NeuralProcessor {
protected:
    double T = 1; //Temperature or steepness of transfer function
public:
    virtual double compute (Input) = 0;
    virtual void initialise() = 0;
};

//Neuron

class Neuron: public NeuralProcessor {
    
    friend std::ostream& operator<<(std::ostream&, const Neuron&);
    friend class NeuronNet;
    friend class NeuralDetector;
    
protected:
    static const int NUMBER_OF_WEIGHTS = 3;
    double w[NUMBER_OF_WEIGHTS];
    //double w0; //weight of threshold, always fires
    
public:
    Neuron() {
        for(int i = 0; i < NUMBER_OF_WEIGHTS; i++) {
            w[i] = 0;
        }
    }
    Neuron(double weight[NUMBER_OF_WEIGHTS]) { //Constructor setting weights of Neuron
        for(int i = 0; i < NUMBER_OF_WEIGHTS; i++) {
            w[i] = weight[i];
        }
	}
    
    void initialise(){ //initialises Neuron weights
		/*
         for(int i = 0; i < NUMBER_OF_WEIGHTS; i++) {
            w[i] = 0;
         }
         */
		
		//Alternatively generate random number between 0 and 1;

        for(int i = 0; i < NUMBER_OF_WEIGHTS; i++) {
            w[i] = ((double) rand() / RAND_MAX);
        }
	}
    
    double compute(Input q){ //computing output from inputs using transfer function
		double a = w[0] + (w[1]*q[0] + w[2]*q[1]);
		double v = (0.5)*(1+tanh(a/T));
		return v;
    }
    
    double computeX(Input q){ //middle step in compute, used to simplify learning algorithm
		double a = w[0] + (w[1]*q[0] + w[2]*q[1]);
		return a;
	}
    
    void update(double weight[NUMBER_OF_WEIGHTS]) { //updating weights of Neuron
        for(int i = 0; i < NUMBER_OF_WEIGHTS; i++) {
            w[i] = weight[i];
        }
	}
    
    void learn(std::vector<Input> Inputmap, std::vector<double> targetmap, double learnrate) {
        //teaches Neuron based on expected Inputs and target using backpropogation
        
        //initialising variables
        double d = 0;
        double errorsum = 1;
        double counter = 0;
        
        if (Inputmap.size() == targetmap.size()) {
            while (std::abs(errorsum) > 0.01) { //while total network error is larger than target (0.01), continue learning
                errorsum = 0;
                for (int c = 0; c < Inputmap.size(); c++ ) {
                    Input a = Inputmap[c];
                    double current = targetmap[c];
                    double target = compute(a);
                    double error = current - target;
                    errorsum = std::abs(error) + errorsum;
                    double deri = (0.5)*pow(cosh(computeX(a)/T),-2)/((double) T);
                    d = learnrate*error*deri;
                    double updated_weights[NUMBER_OF_WEIGHTS] = {w[0]+d, w[1]+a[0]*d, w[2]+a[1]*d};
                    update(updated_weights);
                }
                counter++;
                
                //Verbose output
                //std::cout << "iteration " << counter << " weights: " << w0 << " " << w1 << " " << w2 << "\n";
                
                if (counter > 10000000) { //Timeout
                    std::stringstream  ss;
                    ss << "Loop broken, over " << (counter-1) << " iterations, Current errorsum: " <<errorsum << std::endl;
                    throw (std::string) ss.str();
                }
            }
        }
        else {
            throw (std::string) "Input and target size don't match";
            }
        
    }
};

//Neuron toString
inline
std::ostream& operator<<(std::ostream &strm, const Neuron &a) {
    return strm << "w0 = " << a.w[0] << ",\tw1 = " << a.w[1] << ",\tw2 = " << a.w[2];
}


class Neuron4: public Neuron { //Specialised hidden layer neuron for GEO2L network taking 4 inputs

    friend std::ostream& operator<<(std::ostream&, const Neuron4&);
    friend class NeuralDetector;
    
protected:
    
    static const int NUMBER_OF_WEIGHTS = 5;
    double w[NUMBER_OF_WEIGHTS];

public:
	
	Neuron4() {
        for(int i = 0; i < NUMBER_OF_WEIGHTS; i++) {
            w[i] = ((double) rand() / RAND_MAX);
        }
    }
	Neuron4(double weight[NUMBER_OF_WEIGHTS]) { //Constructor setting weights of Neuron
        for(int i = 0; i < NUMBER_OF_WEIGHTS; i++) {
            w[i] = weight[i];
        }
	}
	
    void initialise(){ //initialises Neuron weights
		/*
         for(int i = 0; i < NUMBER_OF_WEIGHTS; i++) {
            w[i] = 0;
         }
         */
		
		//Alternatively generate random number between 0 and 1;
		
        //srand (static_cast<unsigned int>(time(0))); //Change seed
        
        for(int i = 0; i < NUMBER_OF_WEIGHTS; i++) {
            w[i] = ((double) rand() / RAND_MAX);
        }
	}
	
	double compute(double x[4]){ //computing output from inputs using transfer function
		double a = w[0] + (w[1]*x[0] + w[2]*x[1] + w[3]*x[2] + w[4]*x[3]);
		double v = (0.5)*(1+tanh(a/T));
		return v;
	}
	
	double computeX(double x[4]){ //middle step in compute, used to simplify learning algorithm
		double a = w[0] + (w[1]*x[0] + w[2]*x[1] + w[3]*x[2] + w[4]*x[3]);
		return a;
	}
	
	void update(double weight[NUMBER_OF_WEIGHTS]) { //updating weights of Neuron
        for(int i = 0; i < NUMBER_OF_WEIGHTS; i++) {
            w[i] = weight[i];
        }
	}
    
};

//Neuron4 toString
inline
std::ostream& operator<<(std::ostream &strm, const Neuron4 &a) {
    return strm << "w0 = " << a.w[0] << ",\tw1 = " << a.w[1] << ",\tw2 = " << a.w[2] << ",\tw3 = " << a.w[3] << ",\tw4 = " << a.w[4];
}

class Neuron6 : public Neuron4 { //Specialised output neuron for GEO2L network taking 6 inputs
    
    friend std::ostream& operator<<(std::ostream&, const Neuron6&);
    friend class NeuralDetector;
    
protected:

    static const int NUMBER_OF_WEIGHTS = 7;
    double w[NUMBER_OF_WEIGHTS];

public:
    
	Neuron6() {
        for(int i = 0; i < NUMBER_OF_WEIGHTS; i++) {
            w[i] = ((double) rand() / RAND_MAX);
        }
    }
	Neuron6(double weight[NUMBER_OF_WEIGHTS]) {
		//Constructor setting weights of Neuron
        for(int i = 0; i < NUMBER_OF_WEIGHTS; i++) {
            w[i] = weight[i];
        }
	}
	
	void initialise(){ //initialises Neuron weights
		/*
         for(int i = 0; i < NUMBER_OF_WEIGHTS; i++) {
             w[i] = 0;
         }
         */
		
		//Alternatively generate random number between 0 and 1;
		
        //srand (static_cast<unsigned int>(time(0))); //Change seed
        
        for(int i = 0; i < NUMBER_OF_WEIGHTS; i++) {
            w[i] = ((double) rand() / RAND_MAX);
        }
	}
	
	double compute(double x[6]){ //computing output from inputs using transfer function
		double a = w[0] + (w[1]*x[0] + w[2]*x[1] + w[3]*x[2] + w[4]*x[3] + w[5]*x[4] + w[6]*x[5]);
		double v = (0.5)*(1+tanh(a/T));
		return v;
	}
	
	double computeX(double x[6]){ //middle step in compute, used to simplify learning algorithm
		double a = w[0] + (w[1]*x[0] + w[2]*x[1] + w[3]*x[2] + w[4]*x[3] + w[5]*x[4] + w[6]*x[5]);
		return a;
	}
	
	void update(double weight[NUMBER_OF_WEIGHTS]) {
		//updating weights of Neuron
        for(int i = 0; i < NUMBER_OF_WEIGHTS; i++) {
            w[i] = weight[i];
        }
	}

};

//Neuron6 toString
inline
std::ostream& operator<<(std::ostream &strm, const Neuron6 &a) {
    return strm << "w0 = " << a.w[0] << ",\tw1 = " << a.w[1] << ",\tw2 = " << a.w[2] << ",\tw3 = " << a.w[3] << ",\tw4 = " << a.w[4] << ",\tw5 = " << a.w[5] << ",\tw6 = " << a.w[6];
}

#endif
