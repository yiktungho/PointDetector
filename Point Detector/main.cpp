//
//  main.cpp
//  Point Detector
//
//  Created by Yik Tung Ho on 27/7/14.
//  Copyright (c) 2014年 YTH. All rights reserved.
//

// constructors and derived classes
#include <iostream>
#include "Neuron.h"
#include "Networks.h"

using namespace std;

void testing(NeuralProcessor * n, vector<Input> map) { //Test programmed Neural system's response
   
    for (vector<Input>::size_type i = 0; i != map.size(); i++) {
        cout << "Input [x1 = " << map[i][0] << ", x2 = " << map[i][1] <<
        "] Output = " << n->compute(map[i]) <<endl;
    }
    
}

vector<Input> BooleanInput() { //Generates the 4 possible boolean Inputs
    Input a0 = {1,1};
    Input a1 = {1,0};
    Input a2 = {0,1};
    Input a3 = {0,0};
    vector<Input> Boolean = {a0,a1,a2,a3};
    return (Boolean);
}

//Boolean demo learning targets
vector<double> ANDtarget = {1,0,0,0};
vector<double> ORtarget = {1,1,1,0};
vector<double> NANDtarget = {0,1,1,1};
vector<double> XORtarget = {0,1,1,0};

//Detector Input and target maps

vector<Input> GEOInput() { //Generates GEO Inputs: evenly distributed (0.1 spacing) points from 0 to 1, for testing and learning
    vector<Input> a;
    double x = 0.0;
    double y = 0.0;
    while (x <= 0.9) {
        while (y <= 0.9) {
            Input a1 {x,y};
            a.push_back(a1);
            y = y+0.1;
        }
        y = 0.0;
        x = x+0.1;
    }
    return a;
}

vector<double> GEOtarget() { //Generates a target for GEO learning (even) for use with GEOInput
    vector<double> a;
    for (int i = 0; i < 100; i++) { //25 100
        if (i == 73 ){ //16 73
            a.push_back(1.0);
        }
        else {
            a.push_back(0.0);
        }
    }
    return a;
}

int main () {
    srand (static_cast<unsigned int>(time(0))); //Change seed
    
    cout << "---Begin Single Neuron Demo---" << endl;
    
    //Generating Input
    vector<Input> Inputmap = BooleanInput();
    
    Neuron AND;
    NeuralProcessor * ANDtester = &AND;
    
    Neuron OR;
    NeuralProcessor * ORtester = &OR;
    
    //Teaching AND to a Neuron and testing
    cout << "Initialising AND Neuron" << endl;
    AND.initialise();
    cout << "Current AND Neuron weights" <<endl;
    cout << AND << endl;
    cout << "Testing initialised AND" << endl;
    testing(ANDtester, Inputmap);
    cout << "Teaching AND to Neuron" << endl;;
    try {
        AND.learn(Inputmap, ANDtarget, 0.1);
    }
    catch (string e)
    {
        cout << "EXCEPTION CAUGHT: " << e << endl;
    }
    cout << AND << endl;
    cout << "Testing AND learning" << endl;
    testing(ANDtester, Inputmap);
    
    //Teaching OR to a Neuron and testing
    cout << "Initialising OR Neuron" << endl;
    OR.initialise();
    cout << "Current OR Neuron weights" << endl;
    cout << OR << endl;
    cout << "Testing initialised OR" << endl;
    testing(ORtester, Inputmap);
    
    cout << "Teaching OR to Neuron" << endl;
    try {
        OR.learn(Inputmap, ORtarget, 0.1);
    }
    catch (string e)
    {
        cout << "EXCEPTION CAUGHT: " << e << endl;
    }
    
    cout << OR << endl;
    
    cout << "Testing OR learning" << endl;
    testing(ORtester, Inputmap);
    
    //Teaching NAND to a Neuron and testing

    Neuron NAND;
    NeuralProcessor * NANDtester = &NAND;
    
    cout << "Initialising NAND Neuron" << endl;
    NAND.initialise();
    cout << "Current NAND Neuron weights" << endl;
    cout << NAND << endl;
    cout << "Testing initialised NAND" << endl;
    testing(NANDtester, Inputmap);
    
    cout << "Teaching NAND to Neuron" << endl;
    try {
        NAND.learn(Inputmap, NANDtarget, 0.1);
    }
    catch (string e)
    {
        cout << "EXCEPTION CAUGHT: " << e << endl;
    }
    
    cout << NAND << endl;
    
    cout << "Testing NAND learning" << endl;
    testing(NANDtester, Inputmap);
    
    cout << "---End Single Neuron Demo---" << endl;

    
    cout << "---Begin Simple Neuron Network Demo---" << endl;
    
    NeuronNet ORnet;
    NeuralProcessor * ORnetTester = &ORnet;
    
    cout << "Initialising OR Neuron Net" << endl;
    ORnet.initialise();
    cout << "Current OR Neuron Net weights" << endl;
    cout << ORnet << endl;
    cout << "Testing initialised OR Net" << endl;
    testing(ORnetTester, Inputmap);
    
    cout << "Teaching OR to Neuron Net" << endl;
    try {
        ORnet.learn(Inputmap, ORtarget, 0.1);
    }
    catch (string e)
    {
        cout << "EXCEPTION CAUGHT: " << e << endl;
    }
    
    cout << ORnet << endl;
    cout << "Testing OR learning" << endl;
    testing(ORnetTester, Inputmap);
    
    NeuronNet XOR;
    NeuralProcessor * XORtester = &XOR;
    
    cout << "Initialising XOR Neuron Net" << endl;
    XOR.initialise();
    cout << "Current XOR Neuron Net weights" << endl;
    cout << XOR << endl;
    cout << "Testing initialised XOR Net" << endl;
    testing(XORtester, Inputmap);
    
    cout << "Teaching XOR to Neuron Net" << endl;
    try {
        XOR.learn(Inputmap, XORtarget, 0.1);
    }
    catch (string e)
    {
        cout << "EXCEPTION CAUGHT: " << e << endl;
    }
    
    cout << XOR << endl;
    cout << "Testing XOR learning" << endl;
    testing(XORtester, Inputmap);
    
    cout << "---End Neuron Network Demo---" << endl;
    
    cout << ("---Begin Detector Demo---") << endl;
    cout << ("---Training 2 Hidden Layer Neural Net to identify {0.7,0.3}---") << endl;
    
    vector<Input> GEOInputmap = GEOInput();
    vector<double> GEOtargetmap = GEOtarget();
    
    NeuralDetector detector;
    NeuralProcessor * detectorTester = &detector;
    detector.initialise();
    
    cout << ("Current GEO Neuron Net weights") << endl;
    cout << detector << endl;
    
    cout << "Teaching GEO to Neuron Net [Please be patient: script will normally timeout if teaching takes too long]" << endl;
    cout << "Because initial weights are random, cannot guarantee convergence into desired weights in learning" << endl;
    cout << "under a reasonable time. Please re-run if script times out (after 10 initialisations) if it fails." << endl;
    
    try {
        detector.learn(GEOInputmap, GEOtargetmap, 0.5);
    }
    catch (string e)
    {
        cout << "EXCEPTION CAUGHT: " << e << endl;
    }
    
    cout << detector << endl;
    
    cout << "Testing GEO learning" << endl;
    cout << "Main test, Result of {0.7,0.3}: " << detector.compute(GEOInputmap[73]) << endl;
    testing(detectorTester, GEOInputmap);
    
    cout << ("---End Detector Demo---") << endl;
    
    return 0; }