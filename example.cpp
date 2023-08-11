#include <iostream>
#include <string>
#include <cmath>
#include <iomanip>
#include <fstream>
#include <vector>
#include <algorithm>
#include <time.h>

#include "matrix.h"

using namespace std;

int nspin;
Matrix< complex<double> > sigmaX(2,2);
Matrix< complex<double> > sigmaY(2,2);
Matrix< complex<double> > sigmaZ(2,2);
Matrix< complex<double> > eye(2,2);
Matrix< complex<double> > *interaction12=new Matrix< complex<double> >;
Matrix< complex<double> > *interaction23=new Matrix< complex<double> >;
Matrix< complex<double> > *Ham=new Matrix< complex<double> >;
double J;

int main(int argc, char *argv[])
{
	// Define the Pauli matrices
	
	sigmaX[0][0]=0;
	sigmaX[0][1]=1;
	sigmaX[1][0]=1;
	sigmaX[1][1]=0;
	
	sigmaY[0][0]=0;
	sigmaY[0][1]=(0,-1);
	sigmaY[1][0]=(0,1);
	sigmaY[1][1]=0;
	
	sigmaZ[0][0]=1;
	sigmaZ[0][1]=0;
	sigmaZ[1][0]=0;
	sigmaZ[1][1]=-1;
	
	eye[0][0]=1;
	eye[0][1]=0;
	eye[1][0]=0;
	eye[1][1]=1;
	
	cout<<"SigmaX\n";
	cout<<sigmaX<<'\n';
	cout<<"SigmaY\n";
	cout<<sigmaY<<'\n';
	cout<<"SigmaZ\n";
	cout<<sigmaZ<<'\n';
	
	// The product of sigmaX and sigmaY is sigmaZ
	cout<<"The product of sigmaX and sigmaY is sigmaZ\n";
	cout<<sigmaX*sigmaY<<'\n';
	
	// Form a three-spin Heisenberg chain with coupling constant J
	
	// interaction term between spins 1 and 2
	*interaction12 = KroneckerProduct(KroneckerProduct(sigmaX, sigmaX), eye);
	
	// interaction term between spins 2 and 3
	*interaction23 = KroneckerProduct(eye, KroneckerProduct(sigmaX, sigmaX));
	
	// total Hamiltonian
	J=0.5;
	*Ham = J*(*interaction12 + *interaction23);
	
	cout<<"Hamiltonian:\n";
	cout<<*Ham<<'\n';
	
	complex<double>* e = new complex<double>[(*Ham).columnsize()];		
	EigHValues(*Ham, e, "hermitian");
	
	cout<<"Eigenvalues of the Hamiltonian:\n";
	for(int ei=0; ei<(*Ham).columnsize(); ei++)
		cout<<e[ei]<<'\n';
}

