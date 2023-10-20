#ifndef MATRIX_H
#define MATRIX_H

#include <iostream>
#include <cstdlib>
#include <cmath>
#include <complex>
#include <limits>

/*
--> rowsize() oraz columnsize() return the sizes of columns and rows
--> matrix determinant is calculated via Gauss elimination
--> matrix inverse calculated by solving A*B=Id
--> MatrixPower uses a fast multiplication algorithm
--> KroneckerProduct returns the tensor product of matrices
*/

using namespace std;

template <class T>
class Matrix{
	unsigned int w,k;
	T **M;
public:
	Matrix()
	{
		w=0;
		k=0;
		M=NULL;
	}
	Matrix(unsigned int w, unsigned int k):w(w),k(k)
	{
		M = new T*[w]; //creating an array of pointers
		for(int i=0;i<w;i++)
			M[i] = new T[k];  //M[w][k]
	}
	
	~Matrix()
	{		
		for(int i=0;i<w;i++)
			delete [] M[i];
			
		delete [] M;
		
		w=0;
		k=0;
		M=NULL;			
	}
	
	int rowsize() const;
	int columnsize() const;
	
	template <class X> friend ostream &operator << (ostream &stream, const Matrix<X> &B);
	template <class X> friend istream &operator >> (istream &stream, Matrix<X> &B);
	
	T*& operator [](unsigned int i);
	
	template <class Y, class X> friend Matrix<Y> operator * (const X& r, const Matrix<Y> &A);
	template <class Y, class X> friend Matrix<Y> operator * (const Matrix<Y> &A, const X& r);
	template <class X> friend Matrix<X> operator * (const Matrix<X> &A, const Matrix<X> &B);
	
	template <class X> friend Matrix<X> operator + (const Matrix<X> &A, const Matrix<X> &B);
	template <class X> friend Matrix<X> operator - (const Matrix<X> &A, const Matrix<X> &B);
	
	Matrix<T>& operator= (const Matrix<T>& A);
	Matrix<T>& operator+= (const Matrix<T>& A);
	Matrix<T>& operator-= (const Matrix<T>& A);
	Matrix<T>& operator*= (const T& r);
	Matrix<T>& operator*= (const Matrix<T>& A);
	
	template <class X> friend const Matrix<X> Transpose(const Matrix<X> &A);
	template <class X> friend bool GaussElimination(Matrix<X>& A);
	template <class X> friend void EigenProblem(const Matrix<X> &A, Matrix<X> &V, X eigenval[], const string sym);
	template <class X> friend void EigenValues(const Matrix<X> &A, X eigenval[], const string sym);
	template <class X> friend void EigenProblem(const Matrix<complex<X> > &A, Matrix<complex<X> > &V, complex<X> eigenval[], const string sym);
	template <class X> friend void EigHValues(const Matrix<complex<X> > &A, complex<X> eigenval[], const string sym);
	template <class X> friend double Det(Matrix<X> &A);
	template <class X> friend const Matrix<X> Inv(const Matrix<X> &A);
	template <class X> friend const Matrix<X> MatrixPower(const Matrix<X> &A, int N);
	template <class X> friend X Tr(const Matrix<X> &A);
	template <class X> friend const Matrix<X> KroneckerProduct(const Matrix<X> &A, const Matrix<X> &B);
	template <class X> friend const Matrix<X> zerosMatrix(const int N,const int M);
	template <class X> friend const Matrix<X> onesMatrix(const int N,const int M);
	template <class X> friend bool IsSingular(const Matrix<X>& A);
	template <class X> friend void SolveLinear(Matrix<X> &A, const vector<X>& b, vector<X>& x);
	
	
private:
	void swap_rows(const unsigned int r1, const unsigned int r2);
	void swap_columns(const unsigned int c1, const unsigned int c2);
	Matrix<T> minor(int row, int col) const;
	void tred2(Matrix<T> &Q, T d[], T e[]) const; //tridiagonal form of a symmetric matrix
	void tred2(T d[], T e[]) const;	
	Matrix<T> LUdcmp(vector<int>& indx, int &d);
};

template<class T>
int Matrix<T>::columnsize() const
{
	return w;
}

template<class T>
int Matrix<T>::rowsize() const
{
	return k;
}

template <class T>
ostream &operator << (ostream &stream, const Matrix<T> &B)
{
	for(int i=0;i<B.w;i++)
	{
		for(int j=0;j<B.k;j++)
		{
			stream<<B.M[i][j]<<' ';
		}
		stream<<endl;
	}
	return stream;
}

template <class T>
istream &operator >> (istream &stream, Matrix<T> &B)
{
	string line;
	double el;
	for(int i=0;i<B.w;i++)
	{
		getline(stream,line);
		istringstream sline (line);
		for(int j=0;j<B.k;j++)
		{
			sline>>el;
			B.M[i][j]=el;
		}
	}
	return stream;
}

template <class Y, class X>
Matrix<Y> operator * (const X& r, const Matrix<Y> &A)
{
	Matrix<Y> C(A.w,A.k);
	for(int i=0;i<C.w;i++)
	{
		for(int j=0;j<C.k;j++)
		{
			C.M[i][j]=A.M[i][j]*r;
		}
	}
	return C;
}

template <class Y, class X>
Matrix<Y> operator * (const Matrix<Y> &A, const X& r)
{
	Matrix<Y> C(A.w,A.k);
	for(int i=0;i<C.w;i++)
	{
		for(int j=0;j<C.k;j++)
		{
			C.M[i][j]=A.M[i][j]*r;
		}
	}
	return C;
}

template <class T>
Matrix<T> operator * (const Matrix<T> &A, const Matrix<T> &B)
{
	Matrix<T> C(A.w,B.k);
	if(A.k!=B.w)
	{
		cout<<"Multiplication error: incompatible matrix dimensions!"<<endl;		
		system("pause");
		return C;
	}
	else 
	{
		for(int i=0;i<C.w;i++)
		{
			for(int j=0;j<C.k;j++)
			{
				C.M[i][j]=0.;
				for(int r=0;r<A.k;r++)
				{
					C.M[i][j]+=(A.M[i][r]*B.M[r][j]);
				}
			}
		}
		return C;
	}
}

template <class T>
Matrix<T> operator + (const Matrix<T> &A, const Matrix<T> &B)
{
	Matrix<T> C(A.w,B.k);
	if((A.w!=B.w)||(A.k!=B.k))
	{
		cout<<"Addition arror: incompatible matrix dimensions!"<<endl;		
		system("pause");
		return C;
	}
	else
	{
			for(int i=0;i<C.w;i++)
			{
				for(int j=0;j<C.k;j++)
				{
					C.M[i][j]=A.M[i][j]+B.M[i][j];
				}
			}
			return C;
	}
}

template <class T>
Matrix<T> operator - (const Matrix<T> &A, const Matrix<T> &B)
{
	Matrix<T> C(A.w,B.k);
	if((A.w!=B.w)||(A.k!=B.k))
	{
		cout<<"Subtraction arror: incompatible matrix dimensions!"<<endl;		
		system("pause");
		return C;
	}
	else
	{
			for(int i=0;i<C.w;i++)
			{
				for(int j=0;j<C.k;j++)
				{
					C.M[i][j]=A.M[i][j]-B.M[i][j];
				}
			}
			return C;
	}
}

template <class T>
T*& Matrix<T>::operator [](unsigned int i)
{
	return *(M+i);
}

template <class T>
Matrix<T>& Matrix<T>::operator= (const Matrix<T>& A)
{
	if(M!=NULL)
	{
		for(int i=0;i<w;i++)
			delete [] M[i];
			
		delete [] M;
		
		w=0;
		k=0;
		M=NULL;	
	}
	
	w=A.w;
	k=A.k;
	
	M = new T*[w];
	for(int i=0;i<w;i++)
		M[i] = new T[k];
		
	for(int i=0;i<w;i++)
		for(int j=0;j<k;j++)
			M[i][j]=A.M[i][j];
			
	return *this;				
}

template <class T>
Matrix<T>& Matrix<T>::operator+= (const Matrix<T>& A)
{
	return *this=*this+A;
}

template <class T>
Matrix<T>& Matrix<T>::operator-= (const Matrix<T>& A)
{
	return *this=*this-A;
}

template <class T>
Matrix<T>& Matrix<T>::operator*= (const T& r)
{
	return *this=*this*r;
}

template <class T>
Matrix<T>& Matrix<T>::operator*= (const Matrix<T>& A)
{
	return *this=*this*A;
}

template <class T>
void Matrix<T>::swap_rows(const unsigned int r1, const unsigned int r2)
{
	for(int i=0;i<k;i++)
		swap(M[r1][i],M[r2][i]);
}

template <class T>
void Matrix<T>::swap_columns(const unsigned int c1, const unsigned int c2)
{
	for(int i=0;i<w;i++)
		swap(M[i][c1],M[i][c2]);
}

template <class T> 
bool GaussElimination(Matrix<T>& A)
{
	if(A.w>A.k)
		cout<<"GaussElimination: more rows than columns!"<<endl;
	for(int k=0;k<A.w;k++)
	{
		int i_max=k;
		for(int i=k;i<A.w;i++)
			if(abs(A[i][k])>abs(A[i_max][k])) i_max=i;
		if(A[i_max][k]==0)
			return false;
		A.swap_rows(k, i_max);
		
		for(int i=k+1;i<A.w;i++)
		{
			for(int j=k+1;j<A.k;j++)
				A[i][j]-=A[k][j]*(A[i][k]/A[k][k]);
			A[i][k]=0.;	
		}	
	}
	return true;
}

template <class T>
const Matrix<T> Transpose(const Matrix<T> &A)
{
	Matrix<T> C(A.k,A.w);
	for(int i=0;i<A.w;i++)
		for(int j=0;j<A.k;j++)
			C.M[j][i]=A.M[i][j];
	return C;		
}


template<class T>
T pythag(const T a, const T b)
{
	T absa=abs(a), absb=abs(b);
	return (absa > absb ? absa*sqrt(1.0+(absb/absa)*(absb/absa)) :
		(absb == 0.0 ? 0.0 : absb*sqrt(1.0+(absa/absb)*(absa/absb))));
}

template<class T>
T sign(const T a, const T b)
{
	return(b>=0 ? abs(a) : -abs(a));
}


template <class T> 
void EigenProblem(const Matrix<T> &A, Matrix<T> &V, T eigenval[], const string sym)
{
	int n=A.w;
	T* e=new T[n];
	if(sym=="symmetric")
	{
		A.tred2(V,eigenval,e);
		
		int m, l, iter, i, k;
		T s,r,p,g,f,dd,c,b;
		const T EPS=numeric_limits<T>::epsilon();
		for(i=1;i<n;i++)
			e[i-1]=e[i];
		e[n-1]=0.;
		for(l=0;l<n;l++)
		{
			iter=0;
			do
			{
				for (m=l;m<n-1;m++)
				{
					dd=abs(eigenval[m])+abs(eigenval[m+1]);
					if(abs(e[m])<=EPS*dd) break;
				}
				if(m!=l)
				{
					if(iter++==30) throw("Too many iterations in tqli");
					g=(eigenval[l+1]-eigenval[l])/(2.*e[l]);
					r=pythag(g,1.);
					g=eigenval[m]-eigenval[l]+e[l]/(g+sign(r,g));
					s=c=1.;
					p=0.;
					for(i=m-1;i>=l;i--)
					{
						f=s*e[i];
						b=c*e[i];
						e[i+1]=(r=pythag(f,g));
						if(r==0.)
						{
							eigenval[i+1]-= p;
							e[m]=0.;
							break;	
						}
						s=f/r;
						c=g/r;
						g=eigenval[i+1]-p;
						r=(eigenval[i]-g)*s+2.*c*b;
						eigenval[i+1]=g+(p=s*r);
						g=c*r-b;
						for (k=0;k<n;k++)
						{
							f=V.M[k][i+1];
							V.M[k][i+1]=s*V.M[k][i]+c*f;
							V.M[k][i]=c*V.M[k][i]-s*f;
						}
					}
				if (r == 0.0 && i >= l) continue;
				eigenval[l] -= p;
				e[l]=g;
				e[m]=0.;
				}
			}
			while(m!=l);
		}
	}
	
	//sorting eigenvalues and eigenvectors
	int kk;
	for (int ii=0;ii<n-1;ii++) 
	{
		double p=eigenval[kk=ii];
		for (int j=ii;j<n;j++)
			if (eigenval[j] >= p) p=eigenval[kk=j];
			if (kk != ii) 
			{
				eigenval[kk]=eigenval[ii];
				eigenval[ii]=p;
				if (V.M != NULL)
					for (int j=0;j<n;j++) 
					{
						p=V.M[j][ii];
						V.M[j][ii]=V.M[j][kk];
						V.M[j][kk]=p;
					}
			}
	}
}

template <class T> 
void EigenValues(const Matrix<T> &A, T eigenval[], const string sym)
{
	int n=A.w;
	T* e=new T[n];
	if(sym=="symmetric")
	{
		A.tred2(eigenval,e);
		
		int m, l, iter, i, k;
		T s,r,p,g,f,dd,c,b;
		const T EPS=numeric_limits<T>::epsilon();
		for(i=1;i<n;i++)
			e[i-1]=e[i];
		e[n-1]=0.;
		for(l=0;l<n;l++)
		{
			iter=0;
			do
			{
				for (m=l;m<n-1;m++)
				{
					dd=abs(eigenval[m])+abs(eigenval[m+1]);
					if(abs(e[m])<=EPS*dd) break;
				}
				if(m!=l)
				{
					if(iter++==30) throw("Too many iterations in tqli");
					g=(eigenval[l+1]-eigenval[l])/(2.*e[l]);
					r=pythag(g,1.);
					g=eigenval[m]-eigenval[l]+e[l]/(g+sign(r,g));
					s=c=1.;
					p=0.;
					for(i=m-1;i>=l;i--)
					{
						f=s*e[i];
						b=c*e[i];
						e[i+1]=(r=pythag(f,g));
						if(r==0.)
						{
							eigenval[i+1]-= p;
							e[m]=0.;
							break;	
						}
						s=f/r;
						c=g/r;
						g=eigenval[i+1]-p;
						r=(eigenval[i]-g)*s+2.*c*b;
						eigenval[i+1]=g+(p=s*r);
						g=c*r-b;
					}
				if (r == 0.0 && i >= l) continue;
				eigenval[l] -= p;
				e[l]=g;
				e[m]=0.;
				}
			}
			while(m!=l);
		}
	}
	
	//sorting eigenvalues
	int kk;
	for (int ii=0;ii<n-1;ii++) 
	{
		double p=eigenval[kk=ii];
		for (int j=ii;j<n;j++)
			if (eigenval[j] >= p) p=eigenval[kk=j];
			if (kk != ii) 
			{
				eigenval[kk]=eigenval[ii];
				eigenval[ii]=p;
			}
	}
}

template <class T> 
double Det(Matrix<T> &A)
{
	Matrix<T>* lu=new Matrix<T>;
	int d;
	vector<int> indx;
	(*lu)=A.LUdcmp(indx, d);
	double dd=double(d);
	for(int i=0;i<A.rowsize();i++)
		dd*=(*lu)[i][i];
	delete lu;
	return dd;
}

template <class T> 
const Matrix<T> Inv(const Matrix<T> &A)
{
	if(A.w!=A.k)
	{
		cout<<"Inv: macierz nie jest kwadratowa!"<<endl;
		system("pause");
	}
	int SIZE=A.w;
	
	Matrix<T> B(SIZE,SIZE); //A inverted
	
	Matrix<T> C(SIZE,SIZE); //copy A to C
	C=A;

	
	Matrix<T> Id(SIZE,SIZE);
	for(int i=0;i<SIZE;i++)
		for(int j=0;j<SIZE;j++)
			{
				if(i==j) Id[i][j]=1.;
				else Id[i][j]=0.;
			}
	
	for(int k=0;k<SIZE;k++) //rozwiazuje 3 uklady rownan elim. Gaussa
	{
		int i_max=k;
		for(int i=k;i<SIZE;i++)
			if(abs(C[i][k])>abs(C[i_max][k])) i_max=i;
		if(C[i_max][k]==0)
		{
			cout<<"Inv: matrix is singular!"<<endl;
			system("pause");
		}
		C.swap_rows(k, i_max);
		Id.swap_rows(k,i_max);
		
		for(int i=k+1;i<SIZE;i++)
		{
			for(int j=0;j<SIZE;j++)
			{
				if(j>k)
					C[i][j]-=C[k][j]*(C[i][k]/C[k][k]);
				Id[i][j]-=Id[k][j]*(C[i][k]/C[k][k]);
			}
			C[i][k]=0.;	
		}	
	}
	
	T sum;
	for(int w=SIZE-1;w>=0;w--)
		for(int k=0;k<SIZE;k++)
		{
			sum=0.;
			for(int i=w+1;i<SIZE;i++)
				sum+=B[i][k]*C[w][i];	
			B[w][k]=(Id[w][k]-sum)/C[w][w];	
		}
	
	return B;	
}

template <class T>
Matrix<T> Matrix<T>::minor(int row, int col) const
{
	Matrix<T> Minor(w-1,k-1);
	
	for(int i=0;i<w;i++)
		for(int j=0;j<k;j++)
		{
			if(i<row&&j<col) Minor[i][j]=M[i][j];
			else if(i<row&&j>col) Minor[i][j-1]=M[i][j];
			else if(i>row&&j<col) Minor[i-1][j]=M[i][j];
			else if(i>row&&j>col) Minor[i-1][j-1]=M[i][j];
		}
	return Minor;	
}


template <class T>
const Matrix<T> MatrixPower(const Matrix<T> &A, int N)
{
	int n=(int)floor(log((double)N)/log(2.))+1;
	int* binary=new int[n];
	
	Matrix<T> W(A.w,A.k); //W=Id
	for(int i=0;i<A.w;i++)
		for(int j=0;j<A.k;j++)
		{
			if(i==j) W[i][j]=1.;
			else W[i][j]=0.;
		}
	
	for(int i=0;i<n;i++) //zapis N binarnie
	{
		binary[i]=N%2;
		N/=2;
	}
	
	for(int i=n-1;i>=0;i--) //szybkie potegowanie
	{
		if(binary[i]) W*=W*A;
		else W*=W;
	}
	
	return W;
}

template <class T>
T Tr(const Matrix<T> &A)
{
	T trace=0.;
	
	if(A.w!=A.k)
	{
		cout<<"Tr: matrix is not square!"<<endl;
		system("pause");
	}
	
	for(int i=0;i<A.w;i++)
		trace+=A.M[i][i];
	return trace;	
}

template <class T>
const Matrix<T> KroneckerProduct(const Matrix<T> &A, const Matrix<T> &B)
{
	Matrix<T> C(A.w*B.w,A.k*B.k);
	
	for(int i=0;i<A.w;i++)
	{
		for(int j=0;j<A.k;j++)
		{
			for(int n=0;n<B.w;n++)
			{
				for(int m=0;m<B.k;m++)
				{
					C.M[B.w*i+n][B.k*j+m]=B.M[n][m]*A.M[i][j];
				}
			}	
		}
	}
	
	return C;
}

template <class T>
void Matrix<T>::tred2(Matrix<T> &Q, T d[], T e[]) const
{
	Q=*this;
	int l,k,j,i;
	int n=w;
	T scale, hh, h, g, f;
	
	for(i=n-1;i>0;i--)
	{
		l=i-1;
		h=scale=0.;
		if(l>0)
		{
			for(k=0;k<i;k++)
				scale+=abs(Q.M[i][k]);
			if(scale==0)                  //skip transformation
				e[i]=Q.M[i][l];
			else
			{
				for(k=0;k<i;k++)
				{
					Q.M[i][k]/=scale;         //use scaled a's for transformation
					h+=Q.M[i][k]*Q.M[i][k];     //form sigma in h
				}   
				f=Q.M[i][l]; 
				g=(f >= 0. ? -sqrt(h) : sqrt(h));
				e[i]=scale*g;
				h-=f*g;                    //now h is equation (11.3.4)
				Q.M[i][l]=f-g;             //store u in row i of Q.M
				f=0.; 
				
				for(j=0;j<i;j++)
				{
					Q.M[j][i]=Q.M[i][j]/h;    //store u/H in column i of Q.M
					g=0.;                    //form an element of A.u in g					
					for(k=0;k<j+1;k++)
						g+= Q.M[j][k]*Q.M[i][k];	
					for(k=j+1;k<i;k++)
						g+= Q.M[k][j]*Q.M[i][k];
					e[j]=g/h;                      //form element of p in temporerily
					f+= e[j]*Q.M[i][j];		        // unused element of e
				} 
				hh=f/(h+h);       //form K, equation (11.3.11)
				for(j=0;j<i;j++)  //form q and store in e overwriting p
				{
					f=Q.M[i][j];
					e[j]=g=e[j]-hh*f;					
					for(k=0;k<j+1;k++) //reduce Q.M, equaton (11.2.13)
						Q.M[j][k]-= (f*e[k]+g*Q.M[i][k]);					
				}
			}	
		}
		else
			e[i]=Q.M[i][l];
		d[i]=h;		
	}
	
	d[0]=0.;
	e[0]=0.;
	
	for(i=0;i<n;i++)  //begin accumulation of transformation matrices
	{
		if(d[i]!=0.)    //this block skipped whe i=0
		{
			for(j=0;j<i;j++)
			{
				g=0.;
				for(k=0;k<i;k++)               //use u and u/H stored in Q.M to form P.Q
					g+= Q.M[i][k]*Q.M[k][j];
				for(k=0;k<i;k++)
					Q.M[k][j]-= g*Q.M[k][i];	
			}	
		}
		d[i]=Q.M[i][i];
		Q.M[i][i]=1.;                //reset row and column of Q to identity matrix
		for(j=0;j<i;j++)                // for next iteration
			Q.M[j][i]=Q.M[i][j]=0.;
	}
	//A'=Q^{-1}*A*Q
}

template <class T>
void Matrix<T>::tred2(T d[], T e[]) const
{
	Matrix<T>* Q=new Matrix<T>;
	*Q=*this;
			
	int l,k,j,i;
	int n=w;
	T scale, hh, h, g, f;
	
	for(i=n-1;i>0;i--)
	{
		l=i-1;
		h=scale=0.;
		if(l>0)
		{
			for(k=0;k<i;k++)
				scale+=abs((*Q).M[i][k]);
			if(scale==0)                  //skip transformation
				e[i]=(*Q).M[i][l];
			else
			{
				for(k=0;k<i;k++)
				{
					(*Q).M[i][k]/=scale;         //use scaled a's for transformation
					h+=(*Q).M[i][k]*(*Q).M[i][k];     //form sigma in h
				}   
				f=(*Q).M[i][l]; 
				g=(f >= 0. ? -sqrt(h) : sqrt(h));
				e[i]=scale*g;
				h-=f*g;                    //now h is equation (11.3.4)
				(*Q).M[i][l]=f-g;             //store u in row i of Q.M
				f=0.; 
				
				for(j=0;j<i;j++)
				{
					g=0.;                    //form an element of A.u in g					
					for(k=0;k<j+1;k++)
						g+= (*Q).M[j][k]*(*Q).M[i][k];	
					for(k=j+1;k<i;k++)
						g+= (*Q).M[k][j]*(*Q).M[i][k];
					e[j]=g/h;                      //form element of p in temporerily
					f+= e[j]*(*Q).M[i][j];		        // unused element of e
				} 
				hh=f/(h+h);       //form K, equation (11.3.11)
				for(j=0;j<i;j++)  //form q and store in e overwriting p
				{
					f=(*Q).M[i][j];
					e[j]=g=e[j]-hh*f;					
					for(k=0;k<j+1;k++) //reduce Q.M, equaton (11.2.13)
						(*Q).M[j][k]-= (f*e[k]+g*(*Q).M[i][k]);					
				}
			}	
		}
		else
			e[i]=(*Q).M[i][l];
		d[i]=h;		
	}
	
	e[0]=0.;
	
	for(i=0;i<n;i++)  //begin accumulation of transformation matrices
	{
		d[i]=(*Q).M[i][i];
		(*Q).M[i][i]=1.;                //reset row and column of Q to identity matrix
		for(j=0;j<i;j++)                // for next iteration
			(*Q).M[j][i]=(*Q).M[i][j]=0.;
	}
	delete Q;
}


template <class T> 
const Matrix<T> zerosMatrix(const int N,const int M)
{
	Matrix<T> A(N,M);
	for(int w=0;w<N;w++)
		for(int k=0;k<M;k++)
			A[w][k]=T(0);
	return A;
}

template <class T> 
const Matrix<T> onesMatrix(const int N,const int M)
{
	Matrix<T> A(N,M);
	for(int w=0;w<N;w++)
		for(int k=0;k<M;k++)
			A[w][k]=T(1);
	return A;
}

template <class T> 
bool IsSingular(const Matrix<T>& A)
{
	const T EPS=numeric_limits<T>::epsilon();
	Matrix<T>* M=new Matrix<T>;
	(*M)=A; 
	if((*M).w>(*M).k)
		(*M)=Transpose((*M));
	for(int k=0;k<(*M).w;k++)
	{
		int icol_max=k;
		int irow_max=k;
		for(int i=k;i<(*M).w;i++)
			if(abs((*M)[i][k])>abs((*M)[icol_max][k])) icol_max=i;
		if(abs((*M)[icol_max][k])<EPS)
		{
			for(int i=k;i<(*M).k;i++)
				if(abs((*M)[k][i])>abs((*M)[k][irow_max])) irow_max=i;
			if(abs((*M)[k][irow_max])<EPS)
			{
				delete M;
				return true;
			}
			(*M).swap_columns(k, irow_max);
		}
		else
			(*M).swap_rows(k, icol_max);
		
		for(int i=k+1;i<(*M).w;i++)
		{
			for(int j=k+1;j<(*M).k;j++)
				(*M)[i][j]-=(*M)[k][j]*((*M)[i][k]/(*M)[k][k]);
			(*M)[i][k]=0.;	
		}	
	}
	delete M;
	return false;
}

template <class T>
Matrix<T> Matrix<T>::LUdcmp(vector<int>& indx, int &d)
{
	Matrix<T> lu(w,k);
	for(int ww=0;ww<w;ww++)
		for(int kk=0;kk<k;kk++)
			lu[ww][kk]=M[ww][kk];
	int n=w;
	indx.resize(n);
	d=1;	
	const T TINY=pow(10.,-40);
	int i,imax,j,kk;
	T big,temp;
	vector<T> vv(n);

	for(i=0;i<n;i++)
	{
		big=T(0.);
		for(j=0;j<n;j++)
			if((temp=abs(lu[i][j]))>big) big=temp;
		if(big==T(0.)) 
			cout<<"Singular matrix in LUdcmp"<<endl;
		vv[i]=T(1.)/big;
	}
	for(kk=0;kk<n;kk++)
	{
		big=T(0.);
		for(i=kk;i<n;i++)
		{
			temp=vv[i]*abs(lu[i][kk]);
			if(temp>big)
			{
				big=temp;
				imax=i;
			}
		}
		if(kk!=imax)
		{
			for(j=0;j<n;j++)
			{
				temp=lu[imax][j];
				lu[imax][j]=lu[kk][j];
				lu[kk][j]=temp;
			}
			d=-d;
			vv[imax]=vv[kk];
		}
		indx[kk]=imax;
		if(lu[kk][kk]==0.) lu[kk][kk]=TINY;
		for(i=kk+1;i<n;i++)
		{
			temp=lu[i][kk]/=lu[kk][kk];
			for(j=kk+1;j<n;j++)
				lu[i][j]-=temp*lu[kk][j];
		}
	}
	return lu;
}

template <class T>
void SolveLinear(Matrix<T> &A, const vector<T>& b, vector<T>& x)
{
	Matrix<T>* lu=new Matrix<T>;
	int d;
	vector<int> indx;
	(*lu)=A.LUdcmp(indx, d);
	
	int n=A.rowsize();
	int i,ii=0,ip,j;
	double sum;
	for(i=0;i<n;i++) x[i]=b[i];
	for(i=0;i<n;i++)
	{
		ip=indx[i];
		sum=x[ip];
		x[ip]=x[i];
		if(ii!=0)
			for(j=ii-1;j<i;j++) sum-=(*lu)[i][j]*x[j];
		else if(sum!=0.)
			ii=i+1;
		x[i]=sum;
	}
	for(i=n-1;i>=0;i--)
	{
		sum=x[i];
		for(j=i+1;j<n;j++) sum-=(*lu)[i][j]*x[j];
		x[i]=sum/(*lu)[i][i];
	}
	delete lu;
}

template <>
template <class T>
class Matrix <complex<T> >
{
	int w, k;
	complex<T> **M;
	
public:
	Matrix(){}
	Matrix(unsigned int w, unsigned int k):w(w),k(k)
	{
		M = new complex<T>*[w];//tworzenie tablicy wskaznikow
		for(int i=0;i<w;i++)
			M[i] = new complex<T>[k];  //M[w][k]
	}
	
	~Matrix()
	{		
		for(int i=0;i<w;i++)
			delete [] M[i];
			
		delete [] M;
		
		w=0;
		k=0;
		M=NULL;			
	}
	
	int rowsize() const;
	int columnsize() const;
	
	template <class X> friend ostream &operator << (ostream &stream, const Matrix<complex<X> > &B);
	template <class X> friend istream &operator >> (istream &stream, Matrix<complex<X> > &B);
	
	complex<T>*& operator [](unsigned int i);
	
	template <class Y, class X> friend Matrix<complex<Y> > operator * (const X& r, const Matrix<complex<Y> > &A);
	template <class Y, class X> friend Matrix<complex<Y> > operator * (const Matrix<complex<Y> > &A, const X& r);
	template <class X> friend Matrix<complex<X> > operator * (const Matrix<complex<X> > &A, const Matrix<complex<X> > &B);
	
	template <class X> friend Matrix<complex<X> > operator + (const Matrix<complex<X> > &A, const Matrix<complex<X> > &B);
	template <class X> friend Matrix<complex<X> > operator - (const Matrix<complex<X> > &A, const Matrix<complex<X> > &B);
	
	Matrix<complex <T> >& operator= (const Matrix<complex <T> >& A);
	Matrix<complex<T> >& operator+= (const Matrix<complex<T> >& A);
	Matrix<complex<T> >& operator-= (const Matrix<complex<T> >& A);
	Matrix<complex<T> >& operator*= (const T& r);
	Matrix<complex<T> >& operator*= (const Matrix<complex<T> >& A);
	
	template <class X> friend const Matrix<complex<X> > Dagger(const Matrix<complex<X> > &A);
	template <class X> friend bool GaussElimination(Matrix<complex<X> >& A);
	template <class X> friend void EigHProblem(const Matrix<complex<X> > &A, Matrix<complex<X> > &V, complex<X> eigenval[], const string sym);
	template <class X> friend void EigHValues(const Matrix<complex<X> > &A, complex<X> eigenval[], const string sym);
	template <class X> friend complex<X> Det(const Matrix<complex<X> > &A);
	template <class X> friend const Matrix<complex<X> > Inv(const Matrix<complex<X> > &A);
	template <class X> friend const Matrix<complex<X> > MatrixPower(const Matrix<complex<X> > &A, int N);
	template <class X> friend complex<X> Tr(const Matrix<complex<X> > &A);
	template <class X> friend const Matrix<complex<X> > KroneckerProduct(const Matrix<complex<X> > &A, const Matrix<complex<X> > &B);
	
private:
	void swap_rows(const unsigned int r1, const unsigned int r2);
	void swap_columns(const unsigned int c1, const unsigned int c2);
	Matrix<complex<T> > minor(int row, int col) const;	
};

template<class T>
int Matrix<complex<T> >::columnsize() const
{
	return w;
}

template<class T>
int Matrix<complex<T> >::rowsize() const
{
	return k;
}


template <class T>
ostream &operator << (ostream &stream, const Matrix<complex<T> > &B)
{
	for(int i=0;i<B.w;i++)
	{
		for(int j=0;j<B.k;j++)
		{
			stream<<B.M[i][j]<<' ';
		}
		stream<<endl;
	}
	return stream;
}

template <class T>
istream &operator >> (istream &stream, Matrix<complex<T> > &B)
{
	string line;
	double el;
	for(int i=0;i<B.w;i++)
	{
		getline(stream,line);
		istringstream sline (line);
		for(int j=0;j<B.k;j++)
		{
			sline>>el;
			B.M[i][j]=el;
		}
	}
	return stream;
}

template <class T>
complex<T>*& Matrix<complex<T> >::operator [](unsigned int i)
{
	return *(M+i);
}

template <class Y, class X>
Matrix<complex<Y> > operator * (const X& r, const Matrix<complex<Y> > &A)
{
	Matrix<complex<Y> > C(A.w,A.k);
	for(int i=0;i<C.w;i++)
	{
		for(int j=0;j<C.k;j++)
		{
			C.M[i][j]=A.M[i][j]*r;
		}
	}
	return C;
}

template <class Y, class X>
Matrix<complex<Y> > operator * (const Matrix<complex<Y> > &A, const X& r)
{
	Matrix<complex<Y> > C(A.w,A.k);
	for(int i=0;i<C.w;i++)
	{
		for(int j=0;j<C.k;j++)
		{
			C.M[i][j]=A.M[i][j]*r;
		}
	}
	return C;
}

template <class T>
Matrix<complex<T> > operator * (const Matrix<complex<T> > &A, const Matrix<complex<T> > &B)
{
	Matrix<complex<T> > C(A.w,B.k);
	if(A.k!=B.w)
	{
		cout<<"Multiplication error: incompatible matrix sizes!"<<endl;		
		system("pause");
		return C;
	}
	else 
	{
		for(int i=0;i<C.w;i++)
		{
			for(int j=0;j<C.k;j++)
			{
				C.M[i][j]=0.;
				for(int r=0;r<A.k;r++)
				{
					C.M[i][j]+=(A.M[i][r]*B.M[r][j]);
				}
			}
		}
		return C;
	}
}

template <class T>
Matrix<complex<T> > operator + (const Matrix<complex<T> > &A, const Matrix<complex<T> > &B)
{
	Matrix<complex<T> > C(A.w,B.k);
	if((A.w!=B.w)||(A.k!=B.k))
	{
		cout<<"Addition error: incompatible matrix sizes!"<<endl;		
		system("pause");
		return C;
	}
	else
	{
			for(int i=0;i<C.w;i++)
			{
				for(int j=0;j<C.k;j++)
				{
					C.M[i][j]=A.M[i][j]+B.M[i][j];
				}
			}
			return C;
	}
}

template <class T>
Matrix<complex<T> > operator - (const Matrix<complex<T> > &A, const Matrix<complex<T> > &B)
{
	Matrix<complex<T> > C(A.w,B.k);
	if((A.w!=B.w)||(A.k!=B.k))
	{
		cout<<"Subtraction error: incompatible matrix sizes!"<<endl;		
		system("pause");
		return C;
	}
	else
	{
			for(int i=0;i<C.w;i++)
			{
				for(int j=0;j<C.k;j++)
				{
					C.M[i][j]=A.M[i][j]-B.M[i][j];
				}
			}
			return C;
	}
}

template <class T>
Matrix<complex <T> >& Matrix<complex <T> >::operator= (const Matrix<complex <T> >& A)
{
	if(M!=NULL)
	{
		for(int i=0;i<w;i++)
			delete [] M[i];
			
		delete [] M;
		
		w=0;
		k=0;
		M=NULL;	
	}
	
	w=A.w;
	k=A.k;
	
	M = new complex<T>*[w];
	for(int i=0;i<w;i++)
		M[i] = new complex<T>[k];
		
	for(int i=0;i<w;i++)
		for(int j=0;j<k;j++)
			M[i][j]=A.M[i][j];
			
	return *this;				
}

template <class T>
Matrix<complex<T> >& Matrix<complex<T> >::operator+= (const Matrix<complex<T> >& A)
{
	return *this=*this+A;
}

template <class T>
Matrix<complex<T> >& Matrix<complex<T> >::operator-= (const Matrix<complex<T> >& A)
{
	return *this=*this-A;
}

template <class T>
Matrix<complex<T> >& Matrix<complex<T> >::operator*= (const T& r)
{
	return *this=*this*r;
}

template <class T>
Matrix<complex<T> >& Matrix<complex<T> >::operator*= (const Matrix<complex<T> >& A)
{
	return *this=*this*A;
}

template <class T> 
const Matrix<complex<T> > Dagger(const Matrix<complex<T> > &A)
{
	Matrix<complex<T> > C(A.k,A.w);
	for(int i=0;i<A.w;i++)
		for(int j=0;j<A.k;j++)
			C.M[j][i]=conj(A.M[i][j]);
	return C;		
}

template <class T>
void Matrix<complex<T> >::swap_rows(const unsigned int r1, const unsigned int r2)
{
	for(int i=0;i<k;i++)
		swap(M[r1][i],M[r2][i]);
}

template <class T>
void Matrix<complex<T> >::swap_columns(const unsigned int c1, const unsigned int c2)
{
	for(int i=0;i<w;i++)
		swap(M[i][c1],M[i][c2]);
}

template <class T> 
bool GaussElimination(Matrix<complex<T> >& A)
{
	if(A.w>A.k)
		cout<<"GaussElimination: matrix has more rows than columns!"<<endl;
	for(int k=0;k<A.w;k++)
	{
		int i_max=k;
		for(int i=k;i<A.w;i++)
			if(abs(A[i][k])>abs(A[i_max][k])) i_max=i;
		if(A[i_max][k]==complex<T>(0.,0.))
			return false;
		A.swap_rows(k, i_max);
		
		for(int i=k+1;i<A.w;i++)
		{
			for(int j=k+1;j<A.k;j++)
				A[i][j]-=A[k][j]*(A[i][k]/A[k][k]);
			A[i][k]=complex<T>(0.,0.);	
		}	
	}
	return true;
}

template <class T> 
complex<T> Det(const Matrix<complex<T> > &A)
{
	T sign=1;
	complex<T> det=1.;
	Matrix<complex<T> > B(A.w,A.k);
	
	for(int i=0;i<A.w;i++)
		for(int j=0;j<A.k;j++)
			B[i][j]=A.M[i][j];
	
	if(B.w!=B.k)
	{
		cout<<"Det: matrix is not square!"<<endl;
		system("pause");
	}
	for(int k=0;k<B.w;k++)
	{
		int i_max=k;
		for(int i=k;i<B.w;i++)
			if(abs(B[i][k])>abs(B[i_max][k])) i_max=i;
		if(B[i_max][k]==(0.,0.))
		{
			return complex<T>(0.,0.);
			break;
		}
		B.swap_rows(k, i_max);
		if(k!=i_max) sign*=-1;
		
		for(int i=k+1;i<B.w;i++)
		{
			for(int j=k+1;j<B.k;j++)
				B[i][j]-=B[k][j]*(B[i][k]/B[k][k]);
			B[i][k]=0.;	
		}	
	}
	
	for(int i=0;i<B.w;i++)
	{
		det*=B[i][i];
	}
	
	return det*sign;
}

template <class T>
const Matrix<complex<T> > Inv(const Matrix<complex<T> > &A)
{
	if(A.w!=A.k)
	{
		cout<<"Inv: matrix is not square!"<<endl;
		system("pause");
	}
	int SIZE=A.w;
	
	Matrix<complex<T> > B(SIZE,SIZE); //A inverted
	
	Matrix<complex<T> > M(SIZE,SIZE); //copy A to M
	for(int i=0;i<SIZE;i++)
		for(int j=0;j<SIZE;j++)
			M[i][j]=A.M[i][j];
	
	Matrix<complex<T> > Id(SIZE,SIZE);
	for(int i=0;i<SIZE;i++)
		for(int j=0;j<SIZE;j++)
			{
				if(i==j) Id[i][j]=complex<T>(1.,0.);
				else Id[i][j]=complex<T>(0.,0.);
			}
	
	for(int k=0;k<SIZE;k++)
	{
		int i_max=k;
		for(int i=k;i<SIZE;i++)
			if(abs(M[i][k])>abs(M[i_max][k])) i_max=i;
		if(M[i_max][k]==complex<T>(0.,0.))
		{
			cout<<"Inv: matrix is singular!"<<endl;
			system("pause");
		}
		M.swap_rows(k, i_max);
		Id.swap_rows(k,i_max);
		
		for(int i=k+1;i<SIZE;i++)
		{
			for(int j=0;j<SIZE;j++)
			{
				if(j>k)
					M[i][j]-=M[k][j]*(M[i][k]/M[k][k]);
				Id[i][j]-=Id[k][j]*(M[i][k]/M[k][k]);	
			}
			M[i][k]=complex<T>(0.,0.);	
		}	
	}
	
	complex<T> sum;
	for(int w=SIZE-1;w>=0;w--)
		for(int k=0;k<SIZE;k++)
		{
			sum=0.;
			for(int i=w+1;i<SIZE;i++)
				sum+=B[i][k]*M[w][i];	
			B[w][k]=((Id[w][k])-sum)/M[w][w];	
		}
	return B;	
}

template <class T>
Matrix<complex<T> > Matrix<complex<T> >::minor(int row, int col) const
{
	Matrix<complex<T> > Minor(w-1,k-1);
	
	for(int i=0;i<w;i++)
		for(int j=0;j<k;j++)
		{
			if(i<row&&j<col) Minor[i][j]=M[i][j];
			else if(i<row&&j>col) Minor[i][j-1]=M[i][j];
			else if(i>row&&j<col) Minor[i-1][j]=M[i][j];
			else if(i>row&&j>col) Minor[i-1][j-1]=M[i][j];
		}
	return Minor;
}


template <class T>
const Matrix<complex<T> > MatrixPower(const Matrix<complex<T> > &A, int N)
{
	int n=(int)floor(log((double)N)/log(2.))+1;
	int* binary=new int[n];
	
	Matrix<complex<T> > W(A.w,A.k); //W=Id
	for(int i=0;i<A.w;i++)
		for(int j=0;j<A.k;j++)
		{
			if(i==j) W[i][j]=complex<T>(1.,0.);
			else W[i][j]=complex<T>(0.,0.);
		}
	
	for(int i=0;i<n;i++) //zapis N binarnie
	{
		binary[i]=N%2;
		N/=2;
	}
	
	for(int i=n-1;i>=0;i--) //szybkie potegowanie
	{
		if(binary[i]) W*=W*A;
		else W*=W;
	}
	
	return W;
}

template <class T>
complex<T> Tr(const Matrix<complex<T> > &A)
{
	complex<T> trace=0.;
	
	if(A.w!=A.k)
	{
		cout<<"Tr: macierz nie jest kwadratowa!"<<endl;
		system("pause");
	}
	
	for(int i=0;i<A.w;i++)
		trace+=A.M[i][i];
		
	return trace;	
}

template <class T>
const Matrix<complex<T> > KroneckerProduct(const Matrix<complex<T> > &A, const Matrix<complex<T> > &B)
{
	Matrix<complex<T> > C(A.w*B.w,A.k*B.k);
	
	for(int i=0;i<A.w;i++)
	{
		for(int j=0;j<A.k;j++)
		{
			for(int n=0;n<B.w;n++)
			{
				for(int m=0;m<B.k;m++)
				{
					C.M[B.w*i+n][B.k*j+m]=B.M[n][m]*A.M[i][j];
				}
			}	
		}
	}
	
	return C;
}

template <class T>
void EigHProblem(const Matrix<complex<T> > &A, Matrix<complex<T> > &V, complex<T> eigenval[], const string herm)
{
	int w=A.rowsize();
	int k=A.columnsize();
	
	T x,y;
	
	if(herm=="hermitian")
	{
		Matrix<T> C(2*w,2*k);
		Matrix<T> W(2*w,2*k);
		T*d=new T[2*w];
		for(int i=0;i<2*w;i++)
			for(int j=0;j<2*k;j++)
			{
				if(i<w && j<k) C.M[i][j]=real(A.M[i][j]);
				else if(i>=w && j<k) C.M[i][j]=imag(A.M[i-w][j]);
				else if(i>=w && j>=k) C.M[i][j]=real(A.M[i-w][j-k]);
				else C.M[i][j]=-imag(A.M[i][j-k]);	
			}
		EigenProblem(C,W,d,"symmetric");	
		for(int i=0;i<2*k;i+=2)
		{
			eigenval[i/2]=d[i];	
			for(int j=0;j<w;j++)
			{
				x=W.M[j][i];
				y=W.M[w+j][i];
				V.M[j][i/2]=complex<T>(x,y);
			}
		}
	}
}


template <class T>
void EigHValues(const Matrix<complex<T> > &A, complex<T> eigenval[], const string herm)
{
	int w=A.rowsize();
	int k=A.columnsize();
	
	if(herm=="hermitian")
	{
		Matrix<T> C(2*w,2*k);
		T*d=new T[2*w];
		for(int i=0;i<2*w;i++)
			for(int j=0;j<2*k;j++)
			{
				if(i<w && j<k) C.M[i][j]=real(A.M[i][j]);
				else if(i>=w && j<k) C.M[i][j]=imag(A.M[i-w][j]);
				else if(i>=w && j>=k) C.M[i][j]=real(A.M[i-w][j-k]);
				else C.M[i][j]=-imag(A.M[i][j-k]);	
			}
		EigenValues(C,d,"symmetric");
		
		for(int i=0;i<2*k;i+=2)
		{
			eigenval[i/2]=d[i];	
		}	
	}	
}

#endif // MATRIX_H

