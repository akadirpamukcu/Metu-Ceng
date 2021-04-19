#include "monster_class.hpp"

monster_class::monster_class(){

}

monster_class::monster_class(std::string name, char symbol, std::string x, std::string y, std::vector<std::string> argv){
    monster_response* m_response = new monster_response;
    monster_message* m_message = new monster_message;
    int *pipe = new int[2];
    m_message->game_over=false;
    m_message->damage=0;
    this->name= name;
    this->symbol = symbol;
    this->x=x;
    this->y=y;
    this->argv=argv;
    this->is_dead=false;
}

monster_class::~monster_class(){

}