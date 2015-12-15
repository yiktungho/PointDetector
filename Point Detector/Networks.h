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
    
	Neuron layer1a;
	Neuron layer1b;
	Neuron layer2;
    
public:
    NeuronNet() {
        Neuron layer1a (0,0,0);
        Neuron layer1b (0,0,0);
        Neuron layer2 (0,0,0);
    }
	NeuronNet(Neuron n1, Neuron n2, Neuron n3) { //Constructor setting neurons of neuron net
		layer1a = n1;
		layer1b = n2;
		layer2 = n3;
	}
    
    NeuronNet(double aw0, double aw1, double aw2, double bw0, double bw1, double bw2,
              double cw0, double cw1, double cw2) { //Constructor setting weights of all Neurons
		layer1a.update(aw0, aw1, aw2);
		layer1b.update(bw0, bw1, bw2);
		layer2.update(cw0,  cw1, cw2);
	}
	
	void initialise() { //Initialises all neurons in net
		layer1a.initialise();
		layer1b.initialise();
		layer2.initialise();
	}
    
    double compute(Input q){ //computing output from inputs using transfer function
		double Input1 = 0;
		double Input2 = 0;
		double output = 0;
		Input1 = layer1a.compute(q);
		Input2 = layer1b.compute(q);
		Input q2 = Input {Input1,Input2};
		output = layer2.compute(q2);
		return output;
	}
    
    double computeX(Input q){ //middle step in compute, used to simplify learning algorithm
		double Input1 = 0;
		double Input2 = 0;
		double output = 0;
		Input1 = layer1a.compute(q);
		Input2 = layer1b.compute(q);
		Input q2 = Input {Input1,Input2};
		output = layer2.computeX(q2);
		return output;
	}
    
    void update(double aw0, double aw1, double aw2, double bw0, double bw1, double bw2,
                double cw0, double cw1, double cw2) { //update weights of Neurons in net
		layer1a.update(aw0, aw1, aw2);
		layer1b.update(bw0, bw1, bw2);
		layer2.update(cw0,  cw1, cw2);
	}
    
    void learn(std::vector<Input> Inputmap, std::vector<double> targetmap, double learnrate) {
		//teaches single hiddne layer MLP Neuron Network (Neuronplus) based on expected Inputs and target
        
		//initialising variables
        
		//Update variables to be stored for momentum
		double d = 0;
		double da = 0;
		double db = 0;
		
		double errorsum = 1;
		double counter = 0;
		double initcounter = 0;
        
		double m = 0.5; //momentum term
		
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
                    double deri2 = (0.5)*pow(cosh(computeX(a)/T),-2)/((double) T);
                    
                    d = learnrate*error*deri2 + m*d;
                    
                    double deri1a = (0.5)*pow(cosh(layer1a.computeX(a)/T),-2)/((double) T);
                    
                    double deri1b = (0.5)*pow(cosh(layer1b.computeX(a)/T),-2)/((double) T);
                    
                    double H = error*deri2;
                    
                    da = learnrate*(H*layer2.w[1])*deri1a + m*da;
                    db = learnrate*(H*layer2.w[2])*deri1b + m*db;
                    
                    //Updating Neuron Weights
                    layer2.update(layer2.w[0]+d, layer2.w[1]+layer1a.compute(a)*d, layer2.w[2]+layer1b.compute(a)*d);
                    layer1a.update(layer1a.w[0]+da, layer1a.w[1]+a[0]*da, layer1a.w[2]+a[1]*da);
                    layer1b.update(layer1b.w[0]+db, layer1b.w[1]+a[0]*db, layer1b.w[2]+a[1]*db);
                    
                }
                counter++;
                //std::cout << "Progress: Iterations taken: " << counter << " Initialisations: " << initcounter << " Error: " << errorsum << std::endl; //Verbose Option
                
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
            std::cout << "Complete: Iterations taken: " << counter << " Initialisations: " << initcounter << " Error: " << errorsum << std::endl;
        }

        else {
            throw (std::string) "Input and target size don't match";
        }
    

    }
};

//NeuronNet toString
inline
std::ostream& operator<<(std::ostream &strm, const NeuronNet &a) {
    return strm << "Layer1a: " << a.layer1a << "\n" << "Layer1b: " << a.layer1b << "\n" << "Layer2: " << a.layer2;
}

class NeuralDetector: public NeuralProcessor { //A neuron net made by connecting 11 neurons together, 6 in layer 1, 4 in layer 2 and 1 output neuron
    
    friend std::ostream& operator<<(std::ostream&, const NeuralDetector&);
    
	//First hidden layer
	Neuron layer1a;
	Neuron layer1b;
	Neuron layer1c;
	Neuron layer1d;
	Neuron layer1e;
	Neuron layer1f;
    
	//Second hidden layer
	Neuron6 layer2a;
	Neuron6 layer2b;
	Neuron6 layer2c;
	Neuron6 layer2d;
    
	//Output layer
	Neuron4 layer3;
	
public:
	NeuralDetector() {
        //First hidden layer
        Neuron layer1a (0,0,0);
        Neuron layer1b (0,0,0);
        Neuron layer1c (0,0,0);
        Neuron layer1d (0,0,0);
        Neuron layer1e (0,0,0);
        Neuron layer1f (0,0,0);
        
        //Second hidden layer
        Neuron6 layer2a (0,0,0,0,0,0,0);
        Neuron6 layer2b (0,0,0,0,0,0,0);
        Neuron6 layer2c (0,0,0,0,0,0,0);
        Neuron6 layer2d (0,0,0,0,0,0,0);
        
        //Output layer
        Neuron4 layer3 (0,0,0,0,0);
    }
	NeuralDetector(Neuron n1, Neuron n2, Neuron n3, Neuron n4, Neuron n5, Neuron n6, Neuron6 n21,
                   Neuron6 n22, Neuron6 n23, Neuron6 n24, Neuron4 n31 ) {
        //Constructor setting neurons of neuron net
		
        layer1a = n1;
		layer1b = n2;
		layer1c = n3;
		layer1d = n4;
		layer1e = n5;
		layer1f = n6;
		
		layer2a = n21;
		layer2b = n22;
		layer2c = n23;
		layer2d = n24;
		
		layer3 = n31;
	}
	
	void initialise() { //Initialises all neurons in net
		layer1a.initialise();
		layer1b.initialise();
		layer1c.initialise();
		layer1d.initialise();
		layer1e.initialise();
		layer1f.initialise();
		
		layer2a.initialise();
		layer2b.initialise();
		layer2c.initialise();
		layer2d.initialise();
		
		layer3.initialise();
	}
	
	double compute(Input q){ //computing output from inputs using transfer function
		
		double Input1 = layer1a.compute(q);
		double Input2 = layer1b.compute(q);
		double Input3 = layer1c.compute(q);
		double Input4 = layer1d.compute(q);
		double Input5 = layer1e.compute(q);
		double Input6 = layer1f.compute(q);
		
		double Hidden1 = layer2a.compute(Input1, Input2, Input3, Input4, Input5, Input6);
		double Hidden2 = layer2b.compute(Input1, Input2, Input3, Input4, Input5, Input6);
		double Hidden3 = layer2c.compute(Input1, Input2, Input3, Input4, Input5, Input6);
		double Hidden4 = layer2d.compute(Input1, Input2, Input3, Input4, Input5, Input6);
		
		double output = layer3.compute(Hidden1, Hidden2, Hidden3, Hidden4);
		
		return output;
	}
	
	double computeX(Input q){ //middle step in compute, used to simplify learning algorithm
		
		double Input1 = layer1a.compute(q);
		double Input2 = layer1b.compute(q);
		double Input3 = layer1c.compute(q);
		double Input4 = layer1d.compute(q);
		double Input5 = layer1e.compute(q);
		double Input6 = layer1f.compute(q);
		
		double Hidden1 = layer2a.compute(Input1, Input2, Input3, Input4, Input5, Input6);
		double Hidden2 = layer2b.compute(Input1, Input2, Input3, Input4, Input5, Input6);
		double Hidden3 = layer2c.compute(Input1, Input2, Input3, Input4, Input5, Input6);
		double Hidden4 = layer2d.compute(Input1, Input2, Input3, Input4, Input5, Input6);
		
		double output = layer3.computeX(Hidden1, Hidden2, Hidden3, Hidden4);
		return output;
	}
	
	double computeX2(Input q, std::string select) { //middle step in compute, used to simplify learning algorithm (for 2nd Hidden layer)
        
		double output;
		
		double Input1 = layer1a.compute(q);
		double Input2 = layer1b.compute(q);
		double Input3 = layer1c.compute(q);
		double Input4 = layer1d.compute(q);
		double Input5 = layer1e.compute(q);
		double Input6 = layer1f.compute(q);
		
		double Hidden1 = layer2a.computeX(Input1, Input2, Input3, Input4, Input5, Input6);
		double Hidden2 = layer2b.computeX(Input1, Input2, Input3, Input4, Input5, Input6);
		double Hidden3 = layer2c.computeX(Input1, Input2, Input3, Input4, Input5, Input6);
		double Hidden4 = layer2d.computeX(Input1, Input2, Input3, Input4, Input5, Input6);
		
		if (select == "a") {
			output = Hidden1;
		}
		else if (select == "b") {
			output = Hidden2;
		}
		else if (select == "c") {
			output = Hidden3;
		}
		else if (select == "d") {
			output = Hidden4;
		}
		else {
			throw (std::string) "Neuron selection invalid";
		}
        
		return output;
	}
	
	
	double compute2(Input q, std::string select) { //Normal output of 2nd hidden layer neurons
		
		double output;
		
		double Input1 = layer1a.compute(q);
		double Input2 = layer1b.compute(q);
		double Input3 = layer1c.compute(q);
		double Input4 = layer1d.compute(q);
		double Input5 = layer1e.compute(q);
		double Input6 = layer1f.compute(q);
		
		double Hidden1 = layer2a.compute(Input1, Input2, Input3, Input4, Input5, Input6);
		double Hidden2 = layer2b.compute(Input1, Input2, Input3, Input4, Input5, Input6);
		double Hidden3 = layer2c.compute(Input1, Input2, Input3, Input4, Input5, Input6);
		double Hidden4 = layer2d.compute(Input1, Input2, Input3, Input4, Input5, Input6);
		
		if (select == "a") {
			output = Hidden1;
		}
		else if (select == "b") {
			output = Hidden2;
		}
		else if (select == "c") {
			output = Hidden3;
		}
		else if (select == "d") {
			output = Hidden4;
		}
		else {
			throw (std::string) "Neuron selection invalid";
		}
        
		return output;
	}
    
    void learn (std::vector<Input> Inputmap, std::vector<double> targetmap, double learnrate) {
		//teaches 2 Hidden Layer MLP Neuron Network based on expected Inputs and target (for x-y plane point detector)
		//For use with NeuronGEO2L network
		//Utilises Backpropagation with momentum for learning
		
		//initialising variables
		double current = 0;
		double error = 0;
		double target = 0;
		double errorsum = 1;
		double counter = 0;
		double counterb = 0;
		double initcounter = 0;
		double m = 0; //Momentum term --> originally set to 0.5, but futher testing indicate turning it off is better, possibly an implementation issue
		
		//Second layer update variables
		double d = 0;
		double da = 0;
		double db = 0;
		double dc = 0;
		double dd = 0;
		
		//First layer update variables
		double da1 = 0;
		double db1 = 0;
		double dc1 = 0;
		double dd1 = 0;
		double de1 = 0;
		double df1 = 0;
		
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
                    d = learnrate*error*deri3 + m*d;
                    
                    double deri2a = (0.5)*pow(cosh(computeX2(a,"a")/T),-2)/((double) T);
                    double deri2b = (0.5)*pow(cosh(computeX2(a,"b")/T),-2)/((double) T);
                    double deri2c = (0.5)*pow(cosh(computeX2(a,"c")/T),-2)/((double) T);
                    double deri2d = (0.5)*pow(cosh(computeX2(a,"d")/T),-2)/((double) T);
                    
                    double deri1a = (0.5)*pow(cosh(layer1a.computeX(a)/T),-2)/((double) T);
                    double deri1b = (0.5)*pow(cosh(layer1b.computeX(a)/T),-2)/((double) T);
                    double deri1c = (0.5)*pow(cosh(layer1c.computeX(a)/T),-2)/((double) T);
                    double deri1d = (0.5)*pow(cosh(layer1d.computeX(a)/T),-2)/((double) T);
                    double deri1e = (0.5)*pow(cosh(layer1e.computeX(a)/T),-2)/((double) T);
                    double deri1f = (0.5)*pow(cosh(layer1f.computeX(a)/T),-2)/((double) T);
                    
                    double H = error*deri3;
                    
                    da = learnrate*(H*layer3.w[1])*deri2a + m*da;
                    db = learnrate*(H*layer3.w[2])*deri2b + m*db;
                    dc = learnrate*(H*layer3.w[3])*deri2c + m*dc;
                    dd = learnrate*(H*layer3.w[4])*deri2d + m*dd;
                    
                    da1 = learnrate*(H*(layer2a.w[1]+layer2b.w[1]+layer2c.w[1]+layer2d.w[1]))*deri1a + m*da1;
                    db1 = learnrate*(H*(layer2a.w[2]+layer2b.w[2]+layer2c.w[2]+layer2d.w[2]))*deri1b + m*db1;
                    dc1 = learnrate*(H*(layer2a.w[3]+layer2b.w[3]+layer2c.w[3]+layer2d.w[3]))*deri1c + m*dc1;
                    dd1 = learnrate*(H*(layer2a.w[4]+layer2b.w[4]+layer2c.w[4]+layer2d.w[4]))*deri1d + m*dd1;
                    de1 = learnrate*(H*(layer2a.w[5]+layer2b.w[5]+layer2c.w[5]+layer2d.w[5]))*deri1e + m*de1;
                    df1 = learnrate*(H*(layer2a.w[6]+layer2b.w[6]+layer2c.w[6]+layer2d.w[6]))*deri1f + m*df1;
                    
                    //Updating Neuron Weights
                    layer3.update(layer3.w[0]+d, layer3.w[1]+compute2(a,"a")*d, layer3.w[2]+compute2(a,"b")*d
                                  , layer3.w[3]+compute2(a,"c")*d, layer3.w[4]+compute2(a,"d")*d);
                    
                    layer2a.update(layer2a.w[0]+da, layer2a.w[1]+layer1a.compute(a)*da, layer2a.w[2]+layer1b.compute(a)*da
                                   , layer2a.w[3]+layer1c.compute(a)*da, layer2a.w[4]+layer1d.compute(a)*da
                                   , layer2a.w[5]+layer1e.compute(a)*da, layer2a.w[6]+layer1f.compute(a)*da);
                    layer2b.update(layer2b.w[0]+db, layer2b.w[1]+layer1a.compute(a)*db, layer2b.w[2]+layer1b.compute(a)*db
                                   , layer2b.w[3]+layer1c.compute(a)*db, layer2b.w[4]+layer1d.compute(a)*db
                                   , layer2b.w[5]+layer1e.compute(a)*db, layer2b.w[6]+layer1f.compute(a)*db);
                    layer2c.update(layer2a.w[0]+da, layer2a.w[1]+layer1a.compute(a)*da, layer2a.w[2]+layer1b.compute(a)*dc
                                   , layer2c.w[3]+layer1c.compute(a)*dc, layer2c.w[4]+layer1d.compute(a)*dc
                                   , layer2c.w[5]+layer1e.compute(a)*dc, layer2c.w[6]+layer1f.compute(a)*dc);
                    layer2d.update(layer2a.w[0]+da, layer2a.w[1]+layer1a.compute(a)*da, layer2a.w[2]+layer1b.compute(a)*dd
                                   , layer2d.w[3]+layer1c.compute(a)*dd, layer2d.w[4]+layer1d.compute(a)*dd
                                   , layer2d.w[5]+layer1e.compute(a)*dd, layer2d.w[6]+layer1f.compute(a)*dd);
                    
                    layer1a.update(layer1a.w[0]+da1, layer1a.w[1]+a[0]*da1, layer1a.w[2]+a[1]*da1);
                    layer1b.update(layer1b.w[0]+db1, layer1b.w[1]+a[0]*db1, layer1b.w[2]+a[1]*db1);
                    layer1c.update(layer1c.w[0]+dc1, layer1c.w[1]+a[0]*dc1, layer1c.w[2]+a[1]*dc1);
                    layer1d.update(layer1d.w[0]+dd1, layer1d.w[1]+a[0]*dd1, layer1d.w[2]+a[1]*dd1);
                    layer1e.update(layer1e.w[0]+de1, layer1e.w[1]+a[0]*de1, layer1e.w[2]+a[1]*de1);
                    layer1f.update(layer1f.w[0]+df1, layer1f.w[1]+a[0]*df1, layer1f.w[2]+a[1]*df1);
                    
                }
                
                //std::cout << "Progress: Iterations taken: " << counter << " Initialisations: " << initcounter << " Error: " << errorsum << std::endl; //Verbose option
                
                if (counter == counterb) {
                    std::cout << "Progress: Iterations taken: " << counter << " Initialisations: " << initcounter << " Error: " << errorsum << std::endl;
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
            std::cout << "Complete: Iterations taken: " << counter << " Initialisations: " << initcounter << " Error: " << errorsum << std::endl;
        }
        
        else {
            throw (std::string) "Input and target size don't match";
        }
    }

};

//NeuralDetector toString
inline
std::ostream& operator<<(std::ostream &strm, const NeuralDetector &a) {
    return strm << "Layer1a: " << a.layer1a << "\n" << "Layer1b: " << a.layer1b << "\n" << "Layer1c: " << a.layer1c
    << "\n" << "Layer1d: " << a.layer1d << "\n" << "Layer1e: " << a.layer1e << "\n" << "Layer1f: " << a.layer1f
    << "\n" << "Layer2a: " << a.layer2a << "\n" << "Layer2b: " << a.layer2b << "\n" << "Layer2c: " << a.layer2c
    << "\n" << "Layer2d: " << a.layer2d << "\n" << "Layer3: " << a.layer3;
} 

#endif
