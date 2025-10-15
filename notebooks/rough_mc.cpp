#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef DEFAULT_SUBSAMPLE
#define DEFAULT_SUBSAMPLE 5
#endif
#ifndef LOCK_SUBSAMPLE
#define LOCK_SUBSAMPLE 0
#endif

struct Args {
  std::string mode = "estimate";      // "estimate" | "mc" | "mc_calibrated"
  std::string csv  = "";
  int   time_col   = 0;
  int   price_col  = 1;
  char  delim      = ',';
  bool  has_header = true;

  int   subsample  = DEFAULT_SUBSAMPLE; // seconds per obs after downsample

  int   win10_sec  = 600;  // 10m
  int   win15_sec  = 900;  // 15m
  int   maxlag10   = 6;    // rho=1..6
  int   maxlag15   = 4;    // rho=1..4

  double c_mult    = 3.0;

  double H       = 0.2;
  int    days    = 50;
  int    replications = 200;
  double dt_euler = 0.5;        // seconds for Euler integrator
  double corr_rho = -0.2;
  double v0=0.02, kappa=1.0, theta=0.02, nu=0.23;

  int prec = 6;
};

static void usage() {
  std::cerr <<
    "rough_mc --mode=estimate --csv=spy_all.csv --time-col=0 --price-col=1 --delim=, --has-header=1\n"
    "rough_mc --mode=mc --H=0.2 --days=50 --replications=200\n"
    "rough_mc --mode=mc_calibrated --csv=spy_all.csv --time-col=0 --price-col=1 --days=50 --replications=200\n";
}

static bool parse_int(const std::string& x, int& v){ try{ v=std::stoi(x); return true;}catch(...){return false;} }
static bool parse_double(const std::string& x, double& v){ try{ v=std::stod(x); return true;}catch(...){return false;} }

static Args parse_args(int argc, char** argv){
  Args a;
  for(int i=1;i<argc;i++){
    std::string s(argv[i]);
    auto eat=[&](const char* key)->const char*{
      size_t n=std::strlen(key); if(s.size()>=n && s.compare(0,n,key)==0) return s.c_str()+n; return nullptr;};
    if(auto v=eat("--mode=")) a.mode=v;
    else if(auto v=eat("--csv=")) a.csv=v;
    else if(auto v=eat("--time-col=")) parse_int(v,a.time_col);
    else if(auto v=eat("--price-col=")) parse_int(v,a.price_col);
    else if(auto v=eat("--delim=")) a.delim=*v;
    else if(auto v=eat("--has-header=")){ int t=1; parse_int(v,t); a.has_header=(t!=0); }
    else if(auto v=eat("--subsample=")) parse_int(v,a.subsample);
    else if(auto v=eat("--win10-sec=")) parse_int(v,a.win10_sec);
    else if(auto v=eat("--win15-sec=")) parse_int(v,a.win15_sec);
    else if(auto v=eat("--maxlag10=")) parse_int(v,a.maxlag10);
    else if(auto v=eat("--maxlag15=")) parse_int(v,a.maxlag15);
    else if(auto v=eat("--c-mult=")) parse_double(v,a.c_mult);
    else if(auto v=eat("--H=")) parse_double(v,a.H);
    else if(auto v=eat("--days=")) parse_int(v,a.days);
    else if(auto v=eat("--replications=")) parse_int(v,a.replications);
    else if(auto v=eat("--dt-euler=")) parse_double(v,a.dt_euler);
    else if(auto v=eat("--rho=")) parse_double(v,a.corr_rho);
    else if(auto v=eat("--v0=")) parse_double(v,a.v0);
    else if(auto v=eat("--kappa=")) parse_double(v,a.kappa);
    else if(auto v=eat("--theta=")) parse_double(v,a.theta);
    else if(auto v=eat("--nu=")) parse_double(v,a.nu);
    else if(auto v=eat("--prec=")) parse_int(v,a.prec);
    else { std::cerr<<"Unknown arg: "<<s<<"\n"; usage(); std::exit(1); }
  }
#if LOCK_SUBSAMPLE
  a.subsample = DEFAULT_SUBSAMPLE;
#endif
  return a;
}

static const int EXCLUDED_YMD[] = {
  // FOMC
  20120125,20120313,20120425,20120620,20120801,20120913,20121024,20121212,
  20130130,20130320,20130501,20130619,20130731,20130918,20131030,20131218,
  20140129,20140319,20140430,20140618,20140730,20140917,20141029,20141217,
  20150128,20150318,20150429,20150617,20150729,20150917,20151028,20151216,
  20160127,20160316,20160427,20160615,20160727,20160921,20161102,20161214,
  20170201,20170315,20170503,20170614,20170726,20170920,20171101,20171213,
  20180131,20180321,20180502,20180613,20180801,20180926,20181108,20181219,
  20190130,20190320,20190501,20190619,20190731,20190918,20191030,20191211,
  20200129,20200429,20200610,20200729,20200916,20201105,20201216,
  20210127,20210317,20210428,20210616,20210728,20210922,20211103,20211215,
  20220126,20220316,20220504,20220615,20220727,20220921,20221102,20221214,
  // short/special/halts
  20130703,20131129,20131224,
  20140703,20141030,20141128,20141224,
  20151127,20151224,
  20161125,
  20170703,20171124,
  20180703,20181123,20181224,
  20190703,20190812,20191129,20191224,
  20200309,20200312,20200316,20200318,20201127,20201224,
  20210505,20221126,20221125
};
static inline bool is_excluded(int ymd){
  for(int x: EXCLUDED_YMD) if(x==ymd) return true;
  return false;
}
static inline bool in_RTH(int sec){ return sec >= 9*3600+30*60 && sec <= 16*3600; }

static inline int parse_ymd_hms(const char* s, int& ymd, int& sec){
  int Y=(s[0]-'0')*1000+(s[1]-'0')*100+(s[2]-'0')*10+(s[3]-'0');
  int m=(s[5]-'0')*10 + (s[6]-'0');
  int d=(s[8]-'0')*10 + (s[9]-'0');
  ymd = Y*10000 + m*100 + d;
  int H=(s[11]-'0')*10 + (s[12]-'0');
  int M=(s[14]-'0')*10 + (s[15]-'0');
  int S=(s[17]-'0')*10 + (s[18]-'0');
  sec = H*3600 + M*60 + S;
  return 1;
}

// ---------------- Theta cache Θρ(H) ----------------
static inline double pow_abs(double x, double p){ return std::pow(std::abs(x), p); }

struct ThetaCache {
  int K = 2001;
  double Hmin = 0.001, Hmax = 0.499, dH = 0.0;
  std::vector<std::array<double,7>> th10; // rho=0..6
  std::vector<std::array<double,5>> th15; // rho=0..4
  ThetaCache(){
    dH = (Hmax - Hmin) / (K-1);
    th10.resize(K); th15.resize(K);
    for(int i=0;i<K;++i){
      double H = Hmin + dH*i;
      double p = 2.0*H + 2.0;
      double den = 2.0*(2.0*H + 1.0)*(2.0*H + 2.0);
      auto Theta_rho = [&](int rho)->double{
        double num = pow_abs(rho+2.0,p) - 4.0*pow_abs(rho+1.0,p) + 6.0*pow_abs((double)rho,p)
                   - 4.0*pow_abs(rho-1.0,p) + pow_abs(rho-2.0,p);
        return num/den;
      };
      for(int r=0;r<=6;++r) th10[i][r] = Theta_rho(r);
      for(int r=0;r<=4;++r) th15[i][r] = Theta_rho(r);
    }
  }
} gTheta;

static std::vector<std::vector<double>>
read_daily_returns_RTH(const Args& a){
  std::ifstream fin(a.csv);
  if(!fin){ std::cerr<<"Cannot open: "<<a.csv<<"\n"; std::exit(2); }
  std::string line;
  if(a.has_header && std::getline(fin,line)){ /* skip header */ }

  const int eps = a.subsample;
  const int open_sec = 9*3600 + 30*60;

  std::vector<std::vector<double>> days;
  int cur_ymd=-1, last_emit_t=-1;
  std::vector<double> day_obs; day_obs.reserve(23400/eps + 8);

  auto flush_day=[&](){
    if(day_obs.size()>=2){
      std::vector<double> r; r.reserve(day_obs.size()-1);
      for(size_t i=1;i<day_obs.size();++i){
        double p0=day_obs[i-1], p1=day_obs[i];
        r.push_back((p0>0 && p1>0)? std::log(p1)-std::log(p0) : 0.0);
      }
      days.push_back(std::move(r));
    }
    day_obs.clear(); last_emit_t=-1;
  };

  while(std::getline(fin,line)){
    if(line.empty()) continue;
    int col=0;
    const char* s=line.c_str(); const char* field=s;
    const char* time_ptr=nullptr; const char* price_ptr=nullptr;
    for(size_t i=0;i<=line.size();++i){
      if(i==line.size() || line[i]==a.delim){
        if(col==a.time_col)  time_ptr=field;
        if(col==a.price_col) price_ptr=field;
        ++col; field = s+i+1;
      }
    }
    if(!time_ptr || !price_ptr) continue;

    int ymd, sec; parse_ymd_hms(time_ptr, ymd, sec);
    if(is_excluded(ymd)) continue;
    if(!in_RTH(sec)) continue;

    if(ymd != cur_ymd){
      if(cur_ymd!=-1) flush_day();
      cur_ymd = ymd; day_obs.clear(); last_emit_t=-1;
    }

    double p = std::strtod(price_ptr,nullptr);
    int rth_t = sec - open_sec;
    if(rth_t>=0 && (rth_t % eps) == 0){
      if(rth_t != last_emit_t){ day_obs.push_back(p); last_emit_t = rth_t; }
      else { if(!day_obs.empty()) day_obs.back() = p; }
    }
  }
  if(cur_ymd!=-1) flush_day();
  return days;
}

struct Sums { std::vector<double> S10, S15; };
static inline void ensure_sizes(Sums& Tot){
  if(Tot.S10.empty()) Tot.S10.assign(1+6, 0.0);
  if(Tot.S15.empty()) Tot.S15.assign(1+4, 0.0);
}

static void accumulate_day_S_for_windows_masked(
    const std::vector<double>& rin, int eps, int w10, int w15, int maxlag10, int maxlag15, double c_mult,
    Sums& Tot,
    std::vector<double>* out_c10=nullptr,
    std::vector<double>* out_dkc10=nullptr,
    std::vector<double>* out_retwin10=nullptr
){
  ensure_sizes(Tot);
  if(rin.size()<4) return;

  double mean_r2=0.0; for(double x: rin) mean_r2 += x*x; mean_r2/=rin.size();
  double thr_r = c_mult * std::sqrt(std::max(1e-16, mean_r2));
  std::vector<uint8_t> keep_r(rin.size(), 1);
  for(size_t i=0;i<rin.size();++i) if(std::abs(rin[i])>thr_r) keep_r[i]=0;

  auto process_window = [&](int w, int maxlag, std::vector<double>& S,
                            std::vector<double>* out_c=nullptr,
                            std::vector<double>* out_dkc=nullptr,
                            std::vector<double>* out_retwin=nullptr)
  {
    const size_t N = rin.size();
    if(N < (size_t)(2*w+1)) return;

    std::vector<long double> ps2(N+1,0.0L), ps1(N+1,0.0L);
    for(size_t i=0;i<N;++i){
      if(keep_r[i]) ps2[i+1] = ps2[i] + (long double)rin[i]*(long double)rin[i];
      else          ps2[i+1] = ps2[i];
      ps1[i+1] = ps1[i] + (long double)rin[i]; // unmasked
    }

    const long double norm = 1.0L / ((long double)w * (long double)eps);
    const size_t M  = N - (size_t)w + 1;
    const size_t Md = (M > (size_t)w) ? (M - (size_t)w) : 0;
    if(Md==0) return;

    std::vector<double> c(M,0.0), dkc(Md,0.0), retwin(M,0.0);
    for(size_t t=0;t<M;++t){
      long double s2 = ps2[t+w]-ps2[t];
      c[t] = (double)(s2*norm);
      long double s1 = ps1[t+w]-ps1[t];
      retwin[t] = (double)s1;
    }
    for(size_t t=0;t<Md;++t) dkc[t] = c[t+w] - c[t];

    double mu=0.0; for(double x: dkc) mu += x; mu/=dkc.size();
    double s2=0.0; for(double x: dkc){ double d=x-mu; s2 += d*d; }
    double sd = std::sqrt( s2 / std::max<size_t>(1, dkc.size()-1) );
    double thr_d = c_mult * sd;
    std::vector<uint8_t> keep_dkc(dkc.size(),1);
    for(size_t i=0;i<dkc.size();++i) if(std::abs(dkc[i])>thr_d) keep_dkc[i]=0;

    size_t kept=0; for(auto b: keep_dkc) kept+=b;
    if(kept < dkc.size()/3) std::fill(keep_dkc.begin(), keep_dkc.end(), 1);

    for(int rho=0; rho<=maxlag; ++rho){
      size_t sh = (size_t)rho * (size_t)w;
      long double acc=0.0L;
      for(size_t t=0; t+sh<dkc.size(); ++t){
        if(keep_dkc[t] && keep_dkc[t+sh]) acc += (long double)dkc[t]*(long double)dkc[t+sh];
      }
      S[(size_t)rho] += (double)acc;
    }

    if(out_c)      *out_c = std::move(c);
    if(out_dkc)    *out_dkc = std::move(dkc);
    if(out_retwin) *out_retwin = std::move(retwin);
  };

  process_window(w10, maxlag10, Tot.S10, out_c10, out_dkc10, out_retwin10);
  process_window(w15, maxlag15, Tot.S15, nullptr, nullptr, nullptr);
}

struct EstimationResult{
  double H_hat=std::numeric_limits<double>::quiet_NaN();
  double R_hat=std::numeric_limits<double>::quiet_NaN();
  double H_10m=std::numeric_limits<double>::quiet_NaN();
  double H_15m=std::numeric_limits<double>::quiet_NaN();
};

static std::pair<double,double> solve_R_for_H(const std::vector<double>& V, const std::vector<double>& theta){
  long double num=0.0L, den=0.0L;
  for(size_t i=0;i<V.size();++i){ num += (long double)theta[i]*(long double)V[i]; den += (long double)theta[i]*(long double)theta[i]; }
  double R=(den>0)? (double)(num/den) : 0.0;
  long double J=0.0L; for(size_t i=0;i<V.size();++i){ long double e=(long double)V[i] - (long double)R*(long double)theta[i]; J += e*e; }
  return {R,(double)J};
}

static EstimationResult estimate_H_grid(const Args& a, const Sums& S, int k10, int k15, double eps){
  auto sumabs = [](const std::vector<double>& v){ long double s=0; for(double x: v) s += std::abs(x); return s; };

  if(sumabs(S.S10)+sumabs(S.S15) < 1e-14){
    EstimationResult e;
    e.H_hat  = a.H;
    e.H_10m  = a.H;
    e.H_15m  = a.H;
    e.R_hat  = 0.0;
    return e;
  }

  const int K = 2001;
  std::vector<double> Hgrid(K), sc10(K), sc15(K);
  double log10 = std::log((double)k10*eps), log15 = std::log((double)k15*eps);
  for(int i=0;i<K;++i){
    double H = 0.001 + (0.499-0.001)*i/(K-1.0);
    Hgrid[i]=H;
    sc10[i] = std::exp((1.0 - 2.0*H) * log10);
    sc15[i] = std::exp((1.0 - 2.0*H) * log15);
  }

  auto fit_for = [&](bool only10, bool only15)->std::pair<double,double>{
    double bestH=Hgrid[0], bestR=0.0, bestJ=std::numeric_limits<double>::max();
    for(int i=0;i<K;++i){
      double H = Hgrid[i];
      std::vector<double> V, th;
      if(only10 || (!only10 && !only15)){
        double sc=sc10[i];
        V.push_back(sc*(S.S10[0] + 2.0*S.S10[1]));
        for(int rho=2; rho<=a.maxlag10; ++rho) V.push_back(sc*S.S10[(size_t)rho]);
        double t0 = gTheta.th10[i][0] + 2.0*gTheta.th10[i][1];
        th.push_back(t0);
        for(int rho=2; rho<=a.maxlag10; ++rho) th.push_back(gTheta.th10[i][rho]);
      }
      if(only15 || (!only10 && !only15)){
        double sc=sc15[i];
        V.push_back(sc*(S.S15[0] + 2.0*S.S15[1]));
        for(int rho=2; rho<=a.maxlag15; ++rho) V.push_back(sc*S.S15[(size_t)rho]);
        double t0 = gTheta.th15[i][0] + 2.0*gTheta.th15[i][1];
        th.push_back(t0);
        for(int rho=2; rho<=a.maxlag15; ++rho) th.push_back(gTheta.th15[i][rho]);
      }
      auto [R,J] = solve_R_for_H(V, th);
      if(J<bestJ){ bestJ=J; bestH=H; bestR=R; }
    }
    return {bestH, bestR};
  };

  EstimationResult out;
  auto HB  = fit_for(false,false); out.H_hat=HB.first; out.R_hat=HB.second;
  auto H10 = fit_for(true,false);  out.H_10m=H10.first;
  auto H15 = fit_for(false,true);  out.H_15m=H15.first;
  return out;
}

struct OLSAcc { long double s1=0, sv=0, svv=0, sy=0, svy=0; bool have_v0=false; double v0=0.0; };

static void accumulate_OLS(const std::vector<double>& c10, const std::vector<double>& dkc10,
                           OLSAcc& A){
  const size_t Md = dkc10.size();
  for(size_t t=0;t<Md;++t){
    double vi = c10[t];
    double yi = dkc10[t];
    A.s1 += 1.0L; A.sv += vi; A.svv += (long double)vi*(long double)vi;
    A.sy += yi;   A.svy += (long double)vi*(long double)yi;
  }
  if(!A.have_v0 && !c10.empty()){ A.v0=c10.front(); A.have_v0=true; }
}

static void accumulate_xi_and_rho(const std::vector<double>& c10, const std::vector<double>& dkc10,
                                  const std::vector<double>& retwin10, int w10, double eps,
                                  double c_mult, double alpha, double beta,
                                  long double& SSE, long double& sumvpos,
                                  long double& Rnum, long double& Rdenx, long double& Rdeny)
{
  if(dkc10.empty()) return;
  double mu=0.0; for(double x: dkc10) mu+=x; mu/=dkc10.size();
  double s2=0.0; for(double x: dkc10){ double d=x-mu; s2+=d*d; }
  double sd = std::sqrt(s2 / std::max<size_t>(1, dkc10.size()-1));
  double thr = c_mult * sd;
  std::vector<uint8_t> keep(dkc10.size(),1);
  for(size_t i=0;i<dkc10.size();++i) if(std::abs(dkc10[i])>thr) keep[i]=0;
  size_t kept=0; for(auto b: keep) kept+=b;
  if(kept < dkc10.size()/3) std::fill(keep.begin(), keep.end(), 1);

  const double Delta = (double)(w10*eps);
  for(size_t t=0;t<dkc10.size();++t){
    if(!keep[t]) continue;
    double vi=c10[t], yi=dkc10[t];
    double e = yi - (alpha + beta*vi);
    SSE += (long double)e*(long double)e;
    if(vi>0) sumvpos += vi;
  }
  for(size_t t=0;t<dkc10.size();++t){
    if(!keep[t]) continue;
    size_t j = t + (size_t)w10; if(j >= retwin10.size()) break;
    double vi = c10[t];
    double sc = std::sqrt(std::max(vi,1e-16)*Delta);
    double x  = retwin10[j] / sc;
    double e  = dkc10[t] - (alpha + beta*vi);
    Rnum  += (long double)x * (long double)e / sc; // xi applied later
    Rdenx += (long double)x * (long double)x;
    Rdeny += (long double)e * (long double)e / ( (long double)sc * (long double)sc );
  }
}

// ---------------- Monte Carlo ----------------
static void choose_nu_from_table(double H, double& nu){
  const double Hs[5]={0.1,0.2,0.3,0.4,0.5};
  const double nus[5]={0.14,0.23,0.31,0.39,0.45};
  double bestd=1e9; int bi=4;
  for(int i=0;i<5;++i){ double d=std::abs(H-Hs[i]); if(d<bestd){bestd=d; bi=i;} }
  nu = nus[bi];
}

struct SimOneDay { std::vector<double> r5; };

static SimOneDay simulate_one_day(const Args& a, std::mt19937_64& rng){
  const double Tsec = 23400.0;
  const int    N    = (int)std::llround(Tsec / a.dt_euler);
  const double dt   = a.dt_euler, sqrt_dt=std::sqrt(dt);

  std::normal_distribution<double> Z(0.0,1.0);
  const double rho=a.corr_rho, rho2=std::sqrt(std::max(0.0,1.0-rho*rho));

  std::vector<double> v(N+1, a.v0), x(N+1, 100.0);

  if(a.H >= 0.499){
    for(int n=1;n<=N;++n){
      double z1=Z(rng), z2=Z(rng);
      double dB=z1*sqrt_dt, dW=rho*z1*sqrt_dt + rho2*z2*sqrt_dt;
      double vn=std::max(v[n-1],1e-12);
      double dv = a.kappa*(a.theta - vn)*dt + a.nu*std::sqrt(vn)*dB;
      v[n] = std::max(1e-12, vn + dv);
      x[n] = x[n-1]*std::exp(std::sqrt(v[n])*dW - 0.5*v[n]*dt);
    }
  } else {
    const int    M     = 32;
    const double Tmem  = 36000.0;
    const double alpha = a.H - 0.5;
    std::vector<double> b(M), w(M), y(M,0.0);
    const double bmin=1.0/Tmem, bmax=(double)M/Tmem;
    for(int m=0;m<M;++m){
      double u=(m+0.5)/M;
      b[m] = bmin * std::exp(std::log(bmax/bmin)*u);
      w[m] = std::pow(b[m], -(alpha+1.0)) / M;
    }
    for(int n=1;n<=N;++n){
      double z1=Z(rng), z2=Z(rng);
      double dB=z1*sqrt_dt, dW=rho*z1*sqrt_dt + rho2*z2*sqrt_dt;
      double vn=std::max(v[n-1],1e-12);
      for(int m=0;m<M;++m) y[m] += -b[m]*y[m]*dt + dB;
      double frac=0.0; for(int m=0;m<M;++m) frac += w[m]*y[m];
      double dv = a.kappa*(a.theta - vn)*dt + a.nu*std::sqrt(vn)*frac;
      v[n] = std::max(1e-12, vn + dv);
      x[n] = x[n-1]*std::exp(std::sqrt(v[n])*dW - 0.5*v[n]*dt);
    }
  }

  const int step = std::max(1, (int)std::lround(5.0 / a.dt_euler));
  std::vector<double> r5; r5.reserve(N/step);
  for(int i=step;i<=N;i+=step){
    double p0=x[i-step], p1=x[i];
    r5.push_back(std::log(p1)-std::log(p0));
  }
  return { std::move(r5) };
}

// ---------------- Modes ----------------
static void run_estimate(const Args& a){
  const int  eps  = a.subsample;
  const int  k10  = std::max(1, a.win10_sec/eps);
  const int  k15  = std::max(1, a.win15_sec/eps);

  std::cout<<"Mode: estimate\n";
  auto days = read_daily_returns_RTH(a);
  size_t total_r = 0; for(auto& d: days) total_r += d.size();
  std::cout<<"Days kept: "<<days.size()<<", total returns: "<<total_r<<"\n";

  Sums Tot;
  for(const auto& r : days) accumulate_day_S_for_windows_masked(r, eps, k10, k15, a.maxlag10, a.maxlag15, a.c_mult, Tot);
  auto est = estimate_H_grid(a, Tot, k10, k15, eps);

  std::cout<<std::fixed<<std::setprecision(a.prec);
  std::cout<<"Jump truncation: c="<<a.c_mult<<"\n";
  std::cout<<"H(10m) = "<<est.H_10m<<"\n";
  std::cout<<"H(15m) = "<<est.H_15m<<"\n";
  std::cout<<"H(combined) = "<<est.H_hat<<",  R_hat = "<<est.R_hat<<"\n";
}

struct Calib { double H,kappa,theta,xi,rho,v0; };

static Calib calibrate_from_csv(const Args& a){
  const int eps=a.subsample;
  const int k10=std::max(1, a.win10_sec/eps);
  const int k15=std::max(1, a.win15_sec/eps);

  auto days = read_daily_returns_RTH(a);
  if(days.empty()){ std::cerr<<"No usable RTH days after exclusions.\n"; std::exit(3); }

  Sums Tot;
  std::vector<double> all_c10, all_dkc10, all_retwin10;
  for(const auto& r : days){
    std::vector<double> c10, dkc10, retwin10;
    accumulate_day_S_for_windows_masked(r, eps, k10, k15, a.maxlag10, a.maxlag15, a.c_mult,
                                        Tot, &c10, &dkc10, &retwin10);
    all_c10.insert(all_c10.end(), c10.begin(), c10.end());
    all_dkc10.insert(all_dkc10.end(), dkc10.begin(), dkc10.end());
    all_retwin10.insert(all_retwin10.end(), retwin10.begin(), retwin10.end());
  }

  auto est = estimate_H_grid(a, Tot, k10, k15, eps);
  double Hhat = est.H_hat;

  // OLS on Δv = α + β v (10m)
  OLSAcc A; accumulate_OLS(all_c10, all_dkc10, A);
  long double denom = A.s1*A.svv - A.sv*A.sv;
  double beta  = (denom!=0)? (double)((A.s1*A.svy - A.sv*A.sy)/denom) : 0.0;
  double alpha = (A.s1!=0)?   (double)((A.sy - (long double)beta*A.sv)/A.s1) : 0.0;
  const double Delta = (double)(k10*eps);
  double kappa = (Delta>0)? -beta/Delta : 0.0;
  double theta = (kappa*Delta!=0)? alpha/(kappa*Delta) : 0.02;
  double v0    = A.have_v0 ? A.v0 : theta;

  // SSE → xi and leverage ρ
  long double SSE=0, sumvpos=0, Rnum=0, Rdenx=0, Rdeny=0;
  accumulate_xi_and_rho(all_c10, all_dkc10, all_retwin10, k10, eps, a.c_mult, alpha, beta,
                        SSE, sumvpos, Rnum, Rdenx, Rdeny);
  double xi = (Delta*sumvpos>0)? std::sqrt( (double)SSE / (Delta*(double)sumvpos) ) : 0.23;

  if(xi>0){ Rnum/=xi; Rdeny/=(xi*xi); }
  double rho = (Rdenx>0 && Rdeny>0)? (double)(Rnum / std::sqrt(Rdenx*Rdeny)) : -0.2;
  rho = std::max(-0.999, std::min(0.999, rho));

  return { Hhat, kappa, theta, xi, rho, v0 };
}

static uint64_t splitmix64(uint64_t x){
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

static void run_mc(Args a){
  if(a.nu<=0) choose_nu_from_table(a.H, a.nu);
  std::cout<<"Mode: mc | H="<<a.H<<", days="<<a.days<<", reps="<<a.replications
           <<", dt="<<a.dt_euler<<"s, nu="<<a.nu<<", kappa="<<a.kappa<<", theta="<<a.theta
           <<", rho="<<a.corr_rho<<", v0="<<a.v0<<"\n";

  const double eps=5.0;
  const int k10 = (int)std::lround(600/eps);
  const int k15 = (int)std::lround(900/eps);

  std::vector<double> Hhats(a.replications, std::numeric_limits<double>::quiet_NaN());

  #ifdef _OPENMP
  #pragma omp parallel for schedule(static)
  #endif
  for(int rep=0; rep<a.replications; ++rep){
    uint64_t base = 0xBADC0FFEEULL
    #ifdef _OPENMP
      ^ (uint64_t)omp_get_thread_num()
    #endif
      ^ (uint64_t)(rep+1)*0x9E3779B97F4A7C15ULL;
    std::mt19937_64 rng(splitmix64(base));

    Sums Tot;
    for(int d=0; d<a.days; ++d){
      auto day = simulate_one_day(a, rng);
      accumulate_day_S_for_windows_masked(day.r5, /*eps=*/5, k10, k15, 6, 4, /*c=*/3.0, Tot);
    }

    auto est = estimate_H_grid(a, Tot, k10, k15, /*eps=*/5);

    #ifdef _OPENMP
    #pragma omp critical
    #endif
    std::cout<<std::fixed<<std::setprecision(4)<<"rep "<<rep+1<<": H_hat="<<est.H_hat<<"\n";

    Hhats[rep] = est.H_hat;
  }

  double mu=0, s2=0; int n=0;
  for(double x: Hhats){ if(std::isfinite(x)){ mu+=x; n++; } }
  mu = (n>0)? mu/n : std::numeric_limits<double>::quiet_NaN();
  for(double x: Hhats){ if(std::isfinite(x)){ double d=x-mu; s2+=d*d; } }
  double sd = (n>1)? std::sqrt(s2/(n-1)) : std::numeric_limits<double>::quiet_NaN();

  std::cout<<std::fixed<<std::setprecision(4)
           <<"MC summary: mean(H_hat)="<<mu<<", sd="<<sd<<", n="<<n<<"\n";
}

static void run_mc_calibrated(Args a){
  if(a.csv.empty()){ std::cerr<<"--csv required for mc_calibrated\n"; std::exit(1); }
  std::cout<<"Mode: mc_calibrated — estimating params from CSV\n";
  Calib C = calibrate_from_csv(a);

  double H_used  = std::max(0.05, std::min(0.49, C.H));
  double kappa   = std::max(1e-6, C.kappa);
  double theta   = std::max(1e-6, C.theta);
  double v0      = std::max(theta, std::max(1e-6, C.v0));
  double rho     = std::max(-0.999, std::min(0.999, C.rho));
  double nu_used = C.xi;
  bool warned=false;

  if(nu_used < 1e-3){ choose_nu_from_table(H_used, nu_used); warned=true; }
  if(warned || C.theta<=0 || C.v0<=0 || C.kappa<=0 || C.xi<1e-3){
    std::cout<<"NOTE: calibration produced near-degenerate params; using safe floors/fallbacks "
             <<"(nu="<<nu_used<<", kappa="<<kappa<<", theta="<<theta<<", v0="<<v0<<")\n";
  }

  a.H = H_used; a.kappa=kappa; a.theta=theta; a.nu=nu_used; a.corr_rho=rho; a.v0=v0;

  std::cout<<std::fixed<<std::setprecision(a.prec)
           <<"Calibrated from CSV:\n"
           <<"  H_hat  = "<<C.H<<"\n"
           <<"  kappa  = "<<C.kappa<<"\n"
           <<"  theta  = "<<C.theta<<"\n"
           <<"  xi     = "<<C.xi<<"  (nu used = "<<nu_used<<")\n"
           <<"  rho    = "<<C.rho<<"\n"
           <<"  v0     = "<<C.v0<<"\n"
           <<"Simulating with H="<<a.H<<", days="<<a.days<<", reps="<<a.replications<<"\n";

  run_mc(a);
}

int main(int argc, char** argv){
  std::ios::sync_with_stdio(false);
  std::cin.tie(nullptr);

  Args a = parse_args(argc, argv);
  if(a.mode=="estimate") run_estimate(a);
  else if(a.mode=="mc") run_mc(a);
  else if(a.mode=="mc_calibrated") run_mc_calibrated(a);
  else { usage(); return 1; }
  return 0;
}
