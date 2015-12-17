//
//  Networks.h
//  Point Detector
//
//  Created by Yik Tung Ho on 2/8/14.
//  Copyright (c) 2014年 YTH. All rights reserved.
//

#ifndef Point_Detector_Networks_h
#define Point_Detector_Networks_h

#include "Neuron.h"

class NeuronNet: public NeuralProcessor { //A neuron net made by connecting 3 neurons together, 2 in layer 1 and 1 output neuron
    
    friend std::ostream& operator<<(std::ostream&, const NeuronNet&);
    
    static const int NUM_NEURONS_LAYER_1 = 2;
	Neuron layer1[NUM_NEURONS_LAYER_1];
	Neuron layer2;
    
public:
    NeuronNet() {}
	
	void initialise() { //Initialises all neurons in net
        for(int i = 0; i < NUM_NEURONS_LAYER_1; i++) {
            layer1[i].initialise();
        }
		layer2.initialise();
	}
    
    double compute(Input q){ //computing output from inputs using transfer function
		double output = 0;
        Input q2;
        for(int i = 0; i < NUM_NEURONS_LAYER_1; i++) {
            q2[i] = layer1[i].compute(q);
        }
		output = layer2.compute(q2);
		return output;
	}
    
    double computeX(Input q){ //middle step in compute, used to simplify learning algorithm
		double output = 0;
        Input q2;
        for(int i = 0; i < NUM_NEURONS_LAYER_1; i++) {
            q2[i] = layer1[i].compute(q);
        }
		output = layer2.computeX(q2);
		return output;
	}
    
    void learn(std::vector<Input> Inputmap, std::vector<double> targetmap, double learnrate) {
		//teaches single hidden layer MLP Neuron Network (Neuronplus) based on expected Inputs and target
        
		//initialising variables
        
		//Update variables
		double d_layer_2 = 0;
        double d_layer_1[NUM_NEURONS_LAYER_1] = {0,0};
		
		double errorsum = 1;
		double counter = 0;
		double initcounter = 0;
        
        double updated_weights_layer_1[NUM_NEURONS_LAYER_1][layer1[0].NUMBER_OF_WEIGHTS];
		
        if (Inputmap.size() == targetmap.size()) {
            while (std::abs(errorsum) > 0.01) { //while total network error is larger than target (0.01), continue learning
                errorsum = 0;
                for (int c = 0; c < Inputmap.size(); c++ ) {
                    Input a = Inputmap[c];
                    double current = targetmap[c];
                    double target = compute(a);
                    double error = current - target;
                    errorsum = std::abs(error) + errorsum;
                    
                    //Calculating Update Terms
                    // Layer 2 Calculations
                    double deri2 = (0.5)*pow(cosh(computeX(a)/T),-2)/((double) T);
                    
                    d_layer_2 = learnrate*error*deri2;
                    
                    double H = error*deri2;
                    double updated_weights_a[3] = {layer2.w[0]+d_layer_2, layer2.w[1]+layer1[0].compute(a)*d_layer_2, layer2.w[2]+layer1[1].compute(a)*d_layer_2};
                    
                    // Layer 1 Calculations
                    double deri1[NUM_NEURONS_LAYER_1] = {0,0};
                    
                    for(int i = 0; i < NUM_NEURONS_LAYER_1; i++) {
                        deri1[i] = (0.5)*pow(cosh(layer1[i].computeX(a)/T),-2)/((double) T);
                        d_layer_1[i] = learnrate*(H*layer2.w[i+1])*deri1[i];
                    }

                    for(int i = 0; i < NUM_NEURONS_LAYER_1; i++) {
                        updated_weights_layer_1[i][0] = layer1[i].w[0]+d_layer_1[i];
                        updated_weights_layer_1[i][1] = layer1[i].w[1]+a[0]*d_layer_1[i];
                        updated_weights_layer_1[i][2] = layer1[i].w[2]+a[1]*d_layer_1[i];
                        layer1[i].update(updated_weights_layer_1[i]);
                    }

                    //Updating Neuron Weights
                    layer2.update(updated_weights_a);
                    
                }
                counter++;
                //std::cout << "Progress: Iterations taken: " << counter << "\tInitialisations: " << initcounter << "\tError: " << errorsum << std::endl; //Verbose Option
                
                if (counter > 1000000) {
                    std::cout << "Loop broken, over " << (counter-1) << " iterations" << std::endl; //Verbose Option
                    std::cout << "Initialising Neuron Network and trying again. Initialisations: " << initcounter << std::endl;
                    initialise(); //Because some random initial weights may never converge
                    counter = 0;
                    initcounter++;
                }
                if (initcounter > 10) { //Timeout function
                    throw (std::string) "Too many initialisations, try again";
                }
                
            }
            std::cout << "Complete: Iterations taken: " << counter << "\tInitialisations: " << initcounter << "\tError: " << errorsum << std::endl;
        }

        else {
            throw (std::string) "Input and target size don't match";
        }
    

    }
};

//NeuronNet toString
inline
std::ostream& operator<<(std::ostream &strm, const NeuronNet &a) {
    return strm << "Layer1[0]: " << a.layer1[0] << "\n" << "Layer1[1]: " << a.layer1[1] << "\n" << "Layer2: " << a.layer2;
}

class NeuralDetector: public NeuralProcessor { //A neuron net made by connecting 11 neurons together, 6 in layer 1, 4 in layer 2 and 1 output neuron
    
    friend std::ostream& operator<<(std::ostream&, const NeuralDetector&);
    
    static const int NUM_NEURONS_LAYER_1 = 6;
    static const int NUM_NEURONS_LAYER_2 = 4;
    
    // Number of weights each neuron has in a layer
    static const int NUM_WEIGHTS_LAYER_1 = 4;
    static const int NUM_WEIGHTS_LAYER_2 = 7;
    
    //First hidden layer
    Neuron layer1[NUM_NEURONS_LAYER_1];
    
	//Second hidden layer
    Neuron6 layer2[NUM_NEURONS_LAYER_2];
    
	//Output layer
	Neuron4 layer3;
	
public:
	NeuralDetector() {}
	
	void initialise() { //Initialises all neurons in net
        for(int i = 0; i < NUM_NEURONS_LAYER_1; i++) {
            layer1[i].initialise();
        }
        for(int i = 0; i < NUM_NEURONS_LAYER_2; i++) {
            layer2[i].initialise();
        }
		layer3.initialise();
	}
	
	double compute(Input q){ //computing output from inputs using transfer function
		
        
        double Input[NUM_NEURONS_LAYER_1];
        for(int i = 0; i < NUM_NEURONS_LAYER_1; i++) {
            Input[i] = layer1[i].compute(q);
        }
		
        double Hidden[NUM_NEURONS_LAYER_2];
        for(int i = 0; i < NUM_NEURONS_LAYER_2; i++) {
            Hidden[i] = layer2[i].compute(Input);
        }
		
		double output = layer3.compute(Hidden);
		
		return output;
	}
	
	double computeX(Input q){ //middle step in compute, used to simplify learning algorithm
		
        double Input[NUM_NEURONS_LAYER_1];
        for(int i = 0; i < NUM_NEURONS_LAYER_1; i++) {
            Input[i] = layer1[i].compute(q);
        }
		
        double Hidden[NUM_NEURONS_LAYER_2];
        for(int i = 0; i < NUM_NEURONS_LAYER_2; i++) {
            Hidden[i] = layer2[i].compute(Input);
        }
		
		double output = layer3.computeX(Hidden);
		return output;
	}
	
	double computeX2(Input q, int select) { //middle step in compute, used to simplify learning algorithm (for 2nd Hidden layer)
        
		double output;
		
        double Input[NUM_NEURONS_LAYER_1];
        for(int i = 0; i < NUM_NEURONS_LAYER_1; i++) {
            Input[i] = layer1[i].compute(q);
        }
        
        double Hidden[NUM_NEURONS_LAYER_2];
        for(int i = 0; i < NUM_NEURONS_LAYER_2; i++) {
            Hidden[i] = layer2[i].computeX(Input);
        }
		
		output = Hidden[select];
        
		return output;
	}
	
	
	double compute2(Input q, int select) { //Normal output of 2nd hidden layer neurons
		
		double output;
		
        double Input[NUM_NEURONS_LAYER_1];
        for(int i = 0; i < NUM_NEURONS_LAYER_1; i++) {
            Input[i] = layer1[i].compute(q);
        }
        
        double Hidden[NUM_NEURONS_LAYER_2];
        for(int i = 0; i < NUM_NEURONS_LAYER_2; i++) {
            Hidden[i] = layer2[i].compute(Input);
        }
		
        output = Hidden[select];
		
		return output;
	}
    
    void learn (std::vector<Input> Inputmap, std::vector<double> targetmap, double learnrate) {
		//teaches 2 Hidden Layer MLP Neuron Network based on expected Inputs and target (for x-y plane point detector)
		//For use with NeuronGEO2L network
		//Utilises Backpropagation for learning
		
		//initialising variables
		double current = 0;
		double error = 0;
		double target = 0;
		double errorsum = 1;
		double counter = 0;
		double counterb = 0;
		double initcounter = 0;
		
		//Second layer update variables
		double d = 0;
		
        double d_layer_2[NUM_NEURONS_LAYER_2];
        
        double updated_weights_layer_2[NUM_NEURONS_LAYER_2][layer2[0].NUMBER_OF_WEIGHTS];
		
		//First layer update variables
        double d_layer_1[NUM_NEURONS_LAYER_1] = {0, 0, 0, 0, 0, 0};
        
        double updated_weights_layer_1[NUM_NEURONS_LAYER_1][layer1[0].NUMBER_OF_WEIGHTS];
		
		double deri3 = 0;
        
        if (Inputmap.size() == targetmap.size()) {
            while (std::abs(errorsum) > 0.1) { //while total network error is larger than target (0.1), continue learning
                errorsum = 0;
                for (int c = 0; c < Inputmap.size(); c++ ) { //May implement learning set length detection so other sets can be used
                    Input a = Inputmap[c];
                    current = targetmap[c];
                    target = compute(a);
                    error = current - target;
                    errorsum = std::abs(error) + errorsum; //Total network error
                    
                    //Calculating terms used in update for backpropogation learning
                    
                    deri3 = (0.5)*pow(cosh(computeX(a)/T),-2)/((double) T);
                    d = learnrate*error*deri3;
                    
                    double deri2[NUM_NEURONS_LAYER_2];
                    for(int i = 0; i < NUM_NEURONS_LAYER_2; i++) {
                        deri2[i] = (0.5)*pow(cosh(computeX2(a,i)/T),-2)/((double) T);
                    }
                    
                    double deri1[NUM_NEURONS_LAYER_1];
                    for(int i = 0; i < NUM_NEURONS_LAYER_1; i++) {
                        deri1[i] = (0.5)*pow(cosh(layer1[i].computeX(a)/T),-2)/((double) T);
                    }
                    
                    double H = error*deri3;
                    
                    for(int i = 0; i < NUM_NEURONS_LAYER_2; i++) {
                        d_layer_2[i] = learnrate*(H*layer3.w[1])*deri2[i];
                    }
                    
                    for(int i = 0; i < NUM_NEURONS_LAYER_1; i++) {
                        d_layer_1[i] = learnrate*(H*(layer2[0].w[i+1]+layer2[1].w[i+1]+layer2[2].w[i+1]+layer2[3].w[i+1]))*deri1[i];
                    }
                    
                    //Updating Neuron Weights
                    double updated_weights_3[5] = {layer3.w[0]+d, layer3.w[1]+compute2(a,0)*d, layer3.w[2]+compute2(a,1)*d
                        , layer3.w[3]+compute2(a,2)*d, layer3.w[4]+compute2(a,3)*d};
                    layer3.update(updated_weights_3);
                    
                    for(int i = 0; i < NUM_NEURONS_LAYER_2; i++) {
                        updated_weights_layer_2[i][0] = layer2[i].w[0]+d_layer_2[i];
                        for(int j = 0; j < NUM_NEURONS_LAYER_1; j++) {
                            updated_weights_layer_2[i][j+1] = layer2[i].w[j+1]+layer1[j].compute(a)*d_layer_2[i];
                        }
                        layer2[i].update(updated_weights_layer_2[i]);
                    }
                    
                    for(int i = 0; i < NUM_NEURONS_LAYER_1; i++) {
                        updated_weights_layer_1[i][0] = layer1[i].w[0]+d_layer_1[i];
                        updated_weights_layer_1[i][1] = layer1[i].w[1]+a[0]*d_layer_1[i];
                        updated_weights_layer_1[i][2] = layer1[i].w[2]+a[1]*d_layer_1[i];
                        layer1[i].update(updated_weights_layer_1[i]);
                    }
                    
                }
                
                //std::cout << "Progress: Iterations taken: " << counter << "\tInitialisations: " << initcounter << "\tError: " << errorsum << std::endl; //Verbose option
                
                if (counter == counterb) {
                    std::cout << "Progress: Iterations taken: " << counter << "\tInitialisations: " << initcounter << "\tError: " << errorsum << std::endl;
                    counterb = counterb + 1000;
                }
                counter++;
                
                if (counter > 50000) {
                    std::cout << "Loop broken, over " << (counter-1) << " iterations" << std::endl;
                    std::cout << "Initialising Neuron Network and trying again" << std::endl;
                    initialise(); //Because some random initial weights may never converge
                    counter = 0;
                    counterb = 0;
                    initcounter++;
                }
                if (initcounter > 10) { //Timeout limit
                    throw (std::string) "Too many initialisations, try again";
                }
                
            }
            std::cout << "Complete: Iterations taken: " << counter << "\tInitialisations: " << initcounter << "\tError: " << errorsum << std::endl;
        }
        
        else {
            throw (std::string) "Input and target size don't match";
        }
    }

};

//NeuralDetector toString
inline
std::ostream& operator<<(std::ostream &strm, const NeuralDetector &a) {
    return strm << "Layer1[0]: " << a.layer1[0] << "\n" << "Layer1[1]: " << a.layer1[1] << "\n" << "Layer1[2]: " << a.layer1[2]
    << "\n" << "Layer1[3]: " << a.layer1[3] << "\n" << "Layer1[4]: " << a.layer1[4] << "\n" << "Layer1[5]: " << a.layer1[5]
    << "\n" << "Layer2[0]: " << a.layer2[0] << "\n" << "Layer2[1]: " << a.layer2[1] << "\n" << "Layer2[2]: " << a.layer2[2]
    << "\n" << "Layer2[3]: " << a.layer2[3] << "\n" << "Layer3: " << a.layer3;
} 

#endif
