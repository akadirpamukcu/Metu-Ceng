
#ifndef monster_class_hpp
#define monster_class_hpp

#include "message.h"
#include <iostream>
#include <vector>
#include <string>
#include <sys/types.h>

class monster_class{
    public:
        monster_response* m_response = new monster_response;
        monster_message* m_message = new monster_message;
        int *pipe;
        std::string name;
        char symbol;
        std::string x,y;
        std::vector<std::string> argv;
        int pid;
        int xy=0;
        bool is_dead;
        monster_class();
        monster_class(std::string name, char symbol, std::string x, std::string y, std::vector<std::string> argv);
        ~monster_class();


};


#endif //monster_class_hpp
