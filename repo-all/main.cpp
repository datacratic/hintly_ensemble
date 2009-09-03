#include "github.h"

vector< Rating > data;
vector< vector<int> > ui(USER_NUM);
vector< float > nu(USER_NUM), ni(ITEM_NUM);
vector< vector< pair<int, float> > > w(ITEM_NUM);
UserList users;
ItemList items;
vector< map<string, float> > uname(USER_NUM);
set<int> I;

void rebuildUserLang(){
     vector< map<string,int> > ulang(USER_NUM);
     for(int i = 0; i < data.size(); ++i){
          int ii = data[i].item;
          int uu = data[i].user;
          for(map<string,float>::iterator k = items[ii].language.begin(); k != items[ii].language.end(); ++k){
               ulang[uu][k->first]++;
          }
     }
     for(int i = 0; i < ulang.size(); ++i){
          if(ulang[i].empty()) continue;
          float zz = 0;
          for(map<string,int>::iterator k = ulang[i].begin(); k != ulang[i].end(); ++k)
               zz += (float)(k->second);
          users[i].language.clear();
          for(map<string,int>::iterator k = ulang[i].begin(); k != ulang[i].end(); ++k)
               users[i].language[k->first] = (float)(k->second) / zz;
     }
}

void rebuildUserRepos(){
     vector< map<string,int> > urepos(USER_NUM);
     for(int i = 0; i < data.size(); ++i){
          int ii = data[i].item;
          int uu = data[i].user;
          for(map<string,float>::iterator k = items[ii].repos.begin(); k != items[ii].repos.end(); ++k){
               urepos[uu][k->first]++;
          }
     }
     for(int i = 0; i < urepos.size(); ++i){
          if(urepos[i].empty()) continue;
          float zz = 0;
          for(map<string,int>::iterator k = urepos[i].begin(); k != urepos[i].end(); ++k)
               zz += (float)(k->second);
          users[i].repos.clear();
          for(map<string,int>::iterator k = urepos[i].begin(); k != urepos[i].end(); ++k)
               users[i].repos[k->first] = (float)(k->second) / zz;
     }
}

void model(){
     for(int i = 0; i < data.size(); ++i){
          int uu = data[i].user;
          int ii = data[i].item;
          for(map<string,float>::iterator k = items[ii].name.begin(); k != items[ii].name.end(); ++k){
               if(uname[uu].find(k->first) == uname[uu].end()) uname[uu][k->first] = 0;
               uname[uu][k->first] += 1;
          }
     }
     for(int i = 0; i < uname.size(); ++i){
          if(uname[i].empty()) continue;
          float zz = 0;
          for(map<string,float>::iterator k = uname[i].begin(); k != uname[i].end(); ++k)
               zz += k->second;
          for(map<string,float>::iterator k = uname[i].begin(); k != uname[i].end(); ++k)
               k->second /= zz;
     }
}

void predict(int u, vector< pair<int,float> > & ret){
     set<int> rated(ui[u].begin(), ui[u].end());
     vector<float> cand(ITEM_NUM, 0);
     for(set<int>::iterator i = I.begin(); i != I.end(); ++i){
          int ii = *i;
          if(rated.find(ii) != rated.end()) continue;
          cand[ii] = (1 + sim(uname[u], items[ii].name))
               * (1 + sim(users[u].repos, items[ii].repos))
               * (1 + 0.2 * sim(users[u].language, items[ii].language));
     }
     for(int i = 0; i < cand.size(); ++i)
          if(cand[i] > 0)
               ret.push_back(make_pair<int,float>(i, cand[i]));
     sort(ret.begin(), ret.end(), GreaterSecond<int,float>);
}

void predictAll(bool train){
     loadData(data,train);
     loadUserItemData(users, items, data);
     model();
     rebuildUserRepos();
     rebuildUserLang();
     for(int i = 0; i < data.size(); ++i){
          int user = data[i].user;
          int item = data[i].item;
          ui[user].push_back(item);
          nu[user]++;
          ni[item]++;
          I.insert(item);
     }
     
     int u;
     map<int,int> test;
     if(train) getTestSet2(test);
     else getTestSet(test);
     string file = "../ret2/results-repoall.txt";
     if(train) file += ".0";
     ofstream out(file.c_str());
     for(map<int,int>::iterator k = test.begin(); k != test.end(); ++k){
          int u = k->first;
          vector< pair<int,float> > ret, retlang;
          predict(u, ret);
          
          out << u << "\t";
          for(int i = 0; i < ret.size() && i < 500; ++i)
               out << ret[i].first << "\t" << ret[i].second / ret[0].second << "\t";
          out << endl;
     }
     out.close();
}

int main(int argc, char ** argv){
     bool train = false;
     if(atoi(argv[1]) == 1) train = true;
     predictAll(train);
     return 0;
}

