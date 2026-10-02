/* Fused CPU float64 residuals, analytic derivatives and sparse factor normals.
   Same variable coordinates, robust grouping, priors and clamp conventions as
   the frozen Torch reference. No truth/query data enter these functions. */
#include <math.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#define API __declspec(dllexport)
#else
#define API
#endif

typedef struct { int n,mode,row; double delta,cost; double *r,*J,*g,*H,*w; } Output;
static void factor(Output *o,int d,int k,const int *idx,const double *r,const double *J,double mask) {
    int a,b,c; double s=0,t,w;
    for(a=0;a<d;a++) s+=r[a]*r[a];
    t=sqrt(1+s/(o->delta*o->delta)); w=mask/t; o->cost+=mask*s/(t+1);
    for(a=0;a<d;a++) {
        if(o->r) o->r[o->row+a]=r[a];
        if(o->w) o->w[o->row+a]=w;
        if(o->mode) for(b=0;b<k;b++) if(idx[b]>=0) {
            double v=J[a*k+b];
            if(o->J) o->J[(o->row+a)*o->n+idx[b]]+=v;
            if(o->g) o->g[idx[b]]+=w*r[a]*v;
            if(o->H) for(c=0;c<k;c++) if(idx[c]>=0)
                o->H[idx[b]*o->n+idx[c]]+=w*v*J[a*k+c];
        }
    }
    o->row+=d;
}
static void initialize(Output *o,int n,int m,int mode,double delta,double *r,double *J,double *g,double *H,double *w) {
    o->n=n;o->row=0;o->cost=0;o->mode=mode;o->delta=delta;
    o->r=r;o->J=J;o->g=g;o->H=H;o->w=w;
    if(J) memset(J,0,sizeof(double)*m*n);
    if(g) memset(g,0,sizeof(double)*n);
    if(H) memset(H,0,sizeof(double)*n*n);
}
static void hat(const double *v,double *K) {
    K[0]=0;K[1]=-v[2];K[2]=v[1];K[3]=v[2];K[4]=0;K[5]=-v[0];K[6]=-v[1];K[7]=v[0];K[8]=0;
}
static void mm(const double *a,const double *b,double *c) {
    int i,j,k;for(i=0;i<3;i++)for(j=0;j<3;j++){double s=0;for(k=0;k<3;k++)s+=a[3*i+k]*b[3*k+j];c[3*i+j]=s;}
}
static void rotation(const double *x,double *R,double *D,int derivative) {
    double v[3],K[9],KK[9],s=0,a,b,da,db,t;int i,j;
    for(i=0;i<3;i++){v[i]=.1*x[i];s+=v[i]*v[i];}
    if(s<1e-8){a=1-s/6+s*s/120;b=.5-s/24+s*s/720;da=-1./6+s/60;db=-1./24+s/360;}
    else{t=sqrt(s);a=sin(t)/t;b=(1-cos(t))/s;da=(t*cos(t)-sin(t))/(2*s*t);db=(t*sin(t)/2-(1-cos(t)))/(s*s);}
    hat(v,K);mm(K,K,KK);for(i=0;i<9;i++)R[i]=(i%4==0)+a*K[i]+b*KK[i];
    if(derivative)for(j=0;j<3;j++){
        double e[3]={0,0,0},E[9],EK[9],KE[9];e[j]=1;hat(e,E);mm(E,K,EK);mm(K,E,KE);
        for(i=0;i<9;i++)D[j*9+i]=.1*(2*v[j]*da*K[i]+a*E[i]+2*v[j]*db*KK[i]+b*(EK[i]+KE[i]));
    }
}
static void poses(const double *x,int nodes,double *R,double *D,int derivative) {
    int i,j;for(j=0;j<9;j++)R[j]=(j%4==0);memset(D,0,27*sizeof(double));
    for(i=1;i<nodes;i++)rotation(x+6*(i-1)+3,R+9*i,D+27*i,derivative);
}

/* kind: 0 sensor; 1 radiometric; 2 BA; 3 SE3. Data are per-view support only.
   integer rows: two endpoints / one frame / camera+landmark / source+target.
   doubles rows: observation / source,target,design5,normalizer / observed_uv /
                 source3,target3,mask. BA tail stores fixed depth. */
API int csn5_factors(int kind,int n,int observations,int nodes,int landmarks,int anchor,
        const int *idata,const double *data,const double *x,int mode,double delta,
        double *r,double *J,double *g,double *H,double *w,double *cost) {
    Output o;int i,j,k,c,m;double *R=0,*D=0,*full=0;
    if(n<=0||observations<0||kind<0||kind>3||delta<=0||mode<0||mode>2)return -1;
    m=kind==0?observations+n/3:kind==1?observations+n:kind==2?2*observations:3*observations;
    initialize(&o,n,m,mode,delta,r,J,g,H,w);
    if(kind>=2){
        R=(double*)malloc(9*nodes*sizeof(double));D=(double*)malloc(27*nodes*sizeof(double));
        if(!R||!D){free(R);free(D);return -2;}
        if(kind==2){full=(double*)malloc((n+1)*sizeof(double));if(!full){free(R);free(D);return -2;}
            for(i=0;i<n+1;i++)full[i]=i==anchor?data[2*observations]:x[i-(i>anchor)];
            poses(full,nodes,R,D,mode);
        }else poses(x,nodes,R,D,mode);
    }
    for(i=0;i<observations;i++) {
        int idx[12];double rr[3]={0,0,0},jj[36]={0};
        if(kind==0){
            int p=idata[2*i],q=idata[2*i+1];double pp[2],qq[2],dx,dy,len,bp=0,bq=0;
            if(p<4){pp[0]=p%2;pp[1]=p/2;}else{pp[0]=x[3*(p-4)];pp[1]=x[3*(p-4)+1];bp=x[3*(p-4)+2];}
            if(q<4){qq[0]=q%2;qq[1]=q/2;}else{qq[0]=x[3*(q-4)];qq[1]=x[3*(q-4)+1];bq=x[3*(q-4)+2];}
            dx=pp[0]-qq[0];dy=pp[1]-qq[1];len=sqrt(dx*dx+dy*dy+1e-12);rr[0]=len+bp+bq-data[i];
            for(j=0;j<3;j++){idx[j]=p<4?-1:3*(p-4)+j;idx[j+3]=q<4?-1:3*(q-4)+j;}
            jj[0]=dx/len;jj[1]=dy/len;jj[2]=1;jj[3]=-dx/len;jj[4]=-dy/len;jj[5]=1;
            factor(&o,1,6,idx,rr,jj,1);
        }else if(kind==1){
            int frame=idata[i],start=1+6*frame;const double *d=data+8*i;
            double h=tanh(x[0]),gamma=exp(.5*h),u=0,clipped,gain,power,base,th=tanh(x[start+5]);
            for(j=0;j<5;j++)u+=d[2+j]*x[start+j];clipped=fmin(8,fmax(-8,u));gain=exp(clipped);power=pow(d[0],gamma);base=gain*power;
            rr[0]=(base+.25*th-d[1])*d[7];idx[0]=0;jj[0]=base*log(d[0])*gamma*.5*(1-h*h)*d[7];
            for(j=0;j<5;j++){idx[j+1]=start+j;jj[j+1]=(u>=-8&&u<=8)?base*d[2+j]*d[7]:0;}
            idx[6]=start+5;jj[6]=.25*(1-th*th)*d[7];factor(&o,1,7,idx,rr,jj,1);
        }else if(kind==2){
            int camera=idata[2*i],point=idata[2*i+1],start=6*(nodes-1)+3*point;
            double world[3],cp[3]={0},dc[27]={0},depth;const double *rot=R+9*camera,*dr=D+27*camera;
            for(j=0;j<3;j++)world[j]=.5*full[start+j]-(camera?.1*full[6*(camera-1)+j]:0);
            for(j=0;j<3;j++)for(k=0;k<3;k++)cp[j]+=rot[3*k+j]*world[k];
            depth=fmax(.25,cp[2]);rr[0]=cp[0]/depth-data[2*i];rr[1]=cp[1]/depth-data[2*i+1];
            for(j=0;j<9;j++){int f=j<6?(camera?6*(camera-1)+j:-1):start+j-6;idx[j]=f<0||f==anchor?-1:f-(f>anchor);}
            if(mode){for(j=0;j<3;j++)for(k=0;k<3;k++){
                dc[j*9+k]=-.1*rot[3*k+j];dc[j*9+6+k]=.5*rot[3*k+j];
                for(c=0;c<3;c++)dc[j*9+3+k]+=dr[9*k+3*c+j]*world[c];
            }
            for(j=0;j<2;j++)for(k=0;k<9;k++)jj[j*9+k]=dc[j*9+k]/depth-(cp[2]>=.25?cp[j]*dc[18+k]/(depth*depth):0);}
            factor(&o,2,9,idx,rr,jj,1);
        }else{
            int p=idata[2*i],q=idata[2*i+1];const double *d=data+7*i,*rp=R+9*p,*rq=R+9*q;
            double world[3]={0},tmp[3];
            for(j=0;j<3;j++){
                for(k=0;k<3;k++)world[j]+=rp[3*j+k]*d[k];
                world[j]+=(p?.1*x[6*(p-1)+j]:0)-(q?.1*x[6*(q-1)+j]:0);
            }
            for(j=0;j<3;j++){for(k=0;k<3;k++)rr[j]+=rq[3*k+j]*world[k];rr[j]-=d[3+j];}
            for(j=0;j<6;j++){idx[j]=p?6*(p-1)+j:-1;idx[6+j]=q?6*(q-1)+j:-1;}
            if(mode)for(c=0;c<3;c++){
                for(j=0;j<3;j++){tmp[j]=0;for(k=0;k<3;k++)tmp[j]+=D[p*27+c*9+j*3+k]*d[k];}
                for(j=0;j<3;j++){
                    jj[j*12+c]=.1*rq[3*c+j];jj[j*12+6+c]=-.1*rq[3*c+j];
                    for(k=0;k<3;k++){jj[j*12+3+c]+=rq[3*k+j]*tmp[k];jj[j*12+9+c]+=D[q*27+c*9+3*k+j]*world[k];}
                }
            }
            factor(&o,3,12,idx,rr,jj,d[6]);
        }
    }
    if(kind<2)for(i=0;i<(kind==0?n/3:n);i++){
        int idx=kind==0?3*i+2:i;double a=kind==0?.03:.01,rr=a*x[idx];factor(&o,1,1,&idx,&rr,&a,1);
    }
    free(R);free(D);free(full);*cost=o.cost;return isfinite(o.cost)?0:1;
}

/* Reusable block Gram features without constructing an n x carriers^2 tensor. */
API void csn5_block_gram(int n,int blocks,int width,const int *ids,const double *V,double *G){
    int i,j,k;memset(G,0,sizeof(double)*blocks*width*width);
    for(i=0;i<n;i++)for(j=0;j<width;j++)for(k=0;k<width;k++)G[ids[i]*width*width+j*width+k]+=V[i*width+j]*V[i*width+k];
}

API int csn5_measured(int n,int count,int ns,const double *Ha,const double *Hb,
        const double *seeds,double tolerance,double *Q,double *Ya,double *Yb){
    int rank=0,next=0,krylov=0,i,j,p,pass;double coeff[8],*v;
    if(count<1||count>8||n<1)return -1;
    v=(double*)malloc(n*sizeof(double));if(!v)return -2;
    while(rank<count&&rank<n){
        double original=0,length=0;
        if(next<ns){for(i=0;i<n;i++)v[i]=seeds[i*ns+next];next++;}
        else{if(krylov>=rank)break;for(i=0;i<n;i++)v[i]=.5*(Ya[i*count+krylov]+Yb[i*count+krylov]);krylov++;}
        for(i=0;i<n;i++)original+=v[i]*v[i];original=sqrt(original);
        for(pass=0;pass<2;pass++){
            for(j=0;j<rank;j++){coeff[j]=0;for(i=0;i<n;i++)coeff[j]+=Q[i*count+j]*v[i];}
            for(i=0;i<n;i++){double s=0;for(j=0;j<rank;j++)s+=Q[i*count+j]*coeff[j];v[i]-=s;}
        }
        for(i=0;i<n;i++)length+=v[i]*v[i];length=sqrt(length);
        if(length<=tolerance*fmax(original,1e-300))continue;
        for(i=0;i<n;i++)Q[i*count+rank]=v[i]/length;
        for(i=0;i<n;i++){
            double a=0,b=0;for(p=0;p<n;p++){a+=Ha[i*n+p]*Q[p*count+rank];b+=Hb[i*n+p]*Q[p*count+rank];}
            Ya[i*count+rank]=a;Yb[i*count+rank]=b;
        }
        rank++;
    }
    free(v);return rank;
}

API void csn5_features(int n,int blocks,int k,int edges,const int *ids,const int *edge,
        const double *ga,const double *gb,const double *e,const double *history,
        const double *Ya,const double *Yb,const double *Q,double *V,double *F){
    int i,j,b,p;double norm[20]={0},scale[8]={0};
    memset(F,0,blocks*436*sizeof(double));
    for(i=0;i<n;i++)for(j=0;j<20;j++){
        double v=j==0?e[i]:j==1?ga[i]-gb[i]:j==2?.5*(ga[i]+gb[i]):j==3?history[i]:
            j<12?(j-4<k?Ya[i*k+j-4]:0):(j-12<k?Yb[i*k+j-12]:0);
        V[i*20+j]=v;norm[j]+=v*v;
    }
    for(j=0;j<20;j++)norm[j]=fmax(sqrt(norm[j]),1.221338669755462e-77);
    for(i=0;i<n;i++)for(j=0;j<20;j++)V[i*20+j]/=norm[j];
    for(i=0;i<n;i++){
        b=ids[i];for(j=0;j<20;j++)for(p=0;p<20;p++)F[b*436+j*20+p]+=V[i*20+j]*V[i*20+p];
        F[b*436+400]+=1./n;F[b*436+401]+=V[i*20]*V[i*20];
    }
    for(i=0;i<edges;i++){F[edge[2*i]*436+402]+=1;F[edge[2*i+1]*436+402]+=1;}
    for(b=0;b<blocks;b++){F[b*436+403]=F[b*436+402]>0;F[b*436+402]/=blocks>1?blocks-1:1;}
    for(j=0;j<k;j++){
        for(i=0;i<n;i++)scale[j]+=.5*(Ya[i*k+j]*Ya[i*k+j]+Yb[i*k+j]*Yb[i*k+j]);
        scale[j]=fmax(sqrt(scale[j]),1.221338669755462e-77);
        for(i=0;i<n;i++){
            double a=Ya[i*k+j]/scale[j],c=Yb[i*k+j]/scale[j];b=ids[i];
            F[b*436+404+4*j]+=Q[i*k+j]*.5*(a+c);F[b*436+405+4*j]+=a*a;
            F[b*436+406+4*j]+=c*c;F[b*436+407+4*j]+=a*c;
        }
    }
}

API void csn5_proposal(int n,int blocks,int count,const int *ids,const double *V,const double *raw,double *out){
    int i,j,k,b;double global[160]={0},norm[8]={0};
    for(b=0;b<blocks;b++)for(j=0;j<count;j++)for(k=0;k<20;k++)global[j*20+k]+=raw[(b*8+j)*41+21+k]/blocks;
    for(i=0;i<n;i++)for(j=0;j<count;j++){
        double z=raw[(ids[i]*8+j)*41],u=exp(fmin(z,20)),v=1+u;
        /* Accurate log1p(u) using the portable CRT's log; no old-CRT log1p. */
        double soft=z>20?z:(v==1?u:log(v)*u/(v-1)),s=soft*V[i*20];
        for(k=0;k<20;k++)s+=V[i*20+k]*(raw[(ids[i]*8+j)*41+1+k]+global[j*20+k]);
        out[i*count+j]=s;norm[j]+=s*s;
    }
    for(j=0;j<count;j++)norm[j]=fmax(sqrt(norm[j]),1.221338669755462e-77);
    for(i=0;i<n;i++)for(j=0;j<count;j++)out[i*count+j]/=norm[j];
}
