/* CPU float64 kernels for measured CSN small spaces. No external BLAS dependency.
   Matrices are row-major; every input dimension and Cholesky pivot is checked. */
#include <math.h>
#include <string.h>
#ifdef _WIN32
#define API __declspec(dllexport)
#else
#define API __attribute__((visibility("default")))
#endif
#define MAX_RANK 64

static int evaluate(int n,const double *Ha,const double *Hb,const double *ba,
                    const double *bb,double t,double *x,double *derivative) {
    double L[MAX_RANK*MAX_RANK],rhs[MAX_RANK],y[MAX_RANK];
    double value,sum; int i,j,k;
    memset(L,0,sizeof(L));
    for(i=0;i<n;i++) {
        rhs[i]=bb[i]+t*(ba[i]-bb[i]);
        for(j=0;j<=i;j++) {
            value=Hb[i*n+j]+t*(Ha[i*n+j]-Hb[i*n+j]);
            for(k=0;k<j;k++) value-=L[i*n+k]*L[j*n+k];
            if(i==j) {
                if(!(value>0.0) || !isfinite(value)) return -2;
                L[i*n+j]=sqrt(value);
            } else L[i*n+j]=value/L[j*n+j];
        }
    }
    for(i=0;i<n;i++) {
        value=rhs[i]; for(j=0;j<i;j++) value-=L[i*n+j]*y[j];
        y[i]=value/L[i*n+i];
    }
    for(i=n-1;i>=0;i--) {
        value=y[i]; for(j=i+1;j<n;j++) value-=L[j*n+i]*x[j];
        x[i]=value/L[i*n+i];
    }
    sum=0.;
    for(i=0;i<n;i++) {
        sum+=2.*(ba[i]-bb[i])*x[i];
        for(j=0;j<n;j++) sum-=x[i]*(Ha[i*n+j]-Hb[i*n+j])*x[j];
    }
    *derivative=sum;
    return 0;
}

API int csn_minimax(int n,const double *Ha,const double *Hb,const double *ba,
                   const double *bb,double *x,double *stats) {
    double d,t=0.,lo=0.,hi=1.,gain=0.,ca=0.,cb=0.;
    int i,j,status,solves=0;
    if(n<1 || n>MAX_RANK || !Ha || !Hb || !ba || !bb || !x || !stats) return -1;
    status=evaluate(n,Ha,Hb,ba,bb,0.,x,&d); solves++; if(status) return status;
    if(d<0.) {
        status=evaluate(n,Ha,Hb,ba,bb,1.,x,&d); solves++; if(status) return status;
        if(d<=0.) t=1.;
        else {
            for(i=0;i<36;i++) {
                t=.5*(lo+hi);
                status=evaluate(n,Ha,Hb,ba,bb,t,x,&d); solves++; if(status) return status;
                if(d>0.) hi=t; else lo=t;
            }
            t=.5*(lo+hi);
        }
    }
    status=evaluate(n,Ha,Hb,ba,bb,t,x,&d); solves++; if(status) return status;
    for(i=0;i<n;i++) {
        gain+=.5*(bb[i]+t*(ba[i]-bb[i]))*x[i];
        ca-=ba[i]*x[i]; cb-=bb[i]*x[i];
        for(j=0;j<n;j++) { ca+=.5*x[i]*Ha[i*n+j]*x[j]; cb+=.5*x[i]*Hb[i*n+j]*x[j]; }
    }
    stats[0]=t; stats[1]=gain; stats[2]=fmax(ca,cb)+gain;
    stats[3]=ca; stats[4]=cb; stats[5]=(double)solves;
    return 0;
}

API int csn_orth(int n,int m,const double *V,double tolerance,double reference,double *Q) {
    int column,i,j,pass,rank=0;
    double normsquared,value,coeff[MAX_RANK];
    if(n<1 || m<0 || m>MAX_RANK || tolerance<0. || reference<0. || !V || !Q) return -1;
    memset(Q,0,(size_t)n*m*sizeof(double));
    for(column=0;column<m;column++) {
        for(i=0;i<n;i++) Q[i*m+rank]=V[i*m+column];
        for(pass=0;pass<2;pass++) {
            for(j=0;j<rank;j++) {
                value=0.; for(i=0;i<n;i++) value+=Q[i*m+j]*Q[i*m+rank];
                coeff[j]=value;
            }
            for(i=0;i<n;i++) for(j=0;j<rank;j++) Q[i*m+rank]-=coeff[j]*Q[i*m+j];
        }
        normsquared=0.;for(i=0;i<n;i++) normsquared+=Q[i*m+rank]*Q[i*m+rank];
        value=sqrt(normsquared);
        if(value>tolerance*reference && isfinite(value)) {
            for(i=0;i<n;i++) Q[i*m+rank]/=value;
            rank++;
        }
    }
    return rank;
}
