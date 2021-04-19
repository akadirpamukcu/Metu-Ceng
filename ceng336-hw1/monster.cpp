#include "logging.h"
#include "message.h"
#include "monster_class.hpp"
#include "monster_class.cpp"

#include <sys/wait.h> 
#include <sys/types.h>
#include <unistd.h>
#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include<sys/socket.h>
int distance(coordinate co1, coordinate co2){
    //(std::cerr << co1.x <<  co2.x << co1.y << co2.y << std::endl << std::endl;
    return ( abs(co1.x - co2.x) + abs(co1.y -co2.y) );
}

/*void attackOrMove( monster_message *message, int range,monster_response* response, int damage){
    if(distance(message->new_position, message->player_coordinate) <= range){
        response->mr_type=mr_attack;
        std::cerr << "at monster damage issss firsttt  : " << damage << std::endl;
        response->mr_content.attack = damage;
        std::cerr << "at monster damage issss : " << response->mr_content.attack << std::endl;

    }
    else{
        response->mr_type=mr_move;
        coordinate *new_cor = new coordinate;
        new_cor->x = message->new_position.x;
        new_cor->y = message->new_position.y;
        int x_diff = message->new_position.x - message->player_coordinate.x;
        int y_diff = message->new_position.y - message->player_coordinate.y;
        if(x_diff < 0 && y_diff <0){
            new_cor->x = new_cor->x+1;
            new_cor->y = new_cor->y+1;
        }
        if(x_diff < 0 && y_diff ==0){
            new_cor->x = new_cor->x+1;
        }
        if(x_diff < 0 && y_diff > 0){
            new_cor->x = new_cor->x+1;
            new_cor->y = new_cor->y-1;
        }
        if(x_diff == 0 && y_diff < 0){
            new_cor->y = new_cor->y+1;
        }
        if(x_diff == 0 && y_diff > 0){
            new_cor->y = new_cor->y-1;
        }
        if(x_diff > 0 && y_diff > 0){
            new_cor->x = new_cor->x-1;
            new_cor->y = new_cor->y-1;
        }
        if(x_diff > 0 && y_diff == 0){
            new_cor->x = new_cor->x-1;
        }
        if(x_diff > 0 && y_diff < 0){
            new_cor->x = new_cor->x-1;
            new_cor->y = new_cor->y+1;
        }
        response->mr_content.move_to = *new_cor;
    }

}*/
int healthCalc(int health, int  damage, int defence){
    int real_damage = damage-defence;
    if( 0 > real_damage){
        return health;
    }
    else if(real_damage > 0){
        return health - real_damage;
    }
    else{
        return health;
    }

}
int main(int argc, char **argv){
    std::string symbol;
    int x,y;
    int health,damage,def,range;
    health = atoi(argv[1]);
    damage = atoi(argv[2]);
    def = atoi(argv[3]);
    range = atoi(argv[4]);
    bool move =false;
    bool die = false;
    //std::cerr <<  " \n       range:    "  << range << "          \n" << std::endl;
    //std::cout << health << damage << def << range << std::endl;
    //std::cin >> health >> damage >> def >> range;
    monster_message* message =  new monster_message;
    monster_response*response = new monster_response;    
    response->mr_type = mr_ready;
    write(1,response,sizeof(monster_response));
    int turn=0;
    while(true){
        //std::cerr << "turn is " << turn <<" and it's pid: " << getpid() << std::endl;
        read(1,message, sizeof(monster_message));
        //std::cerr << "At monster. Message received as :: coming damage : "<< message->damage << "position x : " << message->new_position.x  << "position y : " << message->new_position.y  << " game over status: "  << message->game_over << std::endl;
        if(message->game_over) {
            break;
        }
        health= healthCalc(health, (message->damage), def);
        //std::cerr << "new health is : " << health << std::endl;
        if( health <= 0){
            response->mr_type = mr_dead;
            response->mr_content.attack=0;
            write(1,response,sizeof(monster_response));
            break;
        }
        if(distance(message->new_position, message->player_coordinate) <= range){
            //std::cerr <<"distance: " << distance(message->new_position, message->player_coordinate) << " range: " << range << std::endl;
            response->mr_type=mr_attack;
            response->mr_content.attack = damage;
        }
        else{
            int x=0;
            int y=0;
            x = message->new_position.x;
            y = message->new_position.y;
            int x_diff = x - message->player_coordinate.x;
            int y_diff = y - message->player_coordinate.y;
            //std::cerr <<"x_diff: " << x_diff<< " y_diff: " << y_diff << std::endl;
            if( x_diff < 0 && y_diff <0 ){
                x = x+1;
                y = y+1;
            }
            else if(x_diff < 0 && y_diff ==0){
                x = x+1;
            }
            else if(x_diff < 0 && y_diff > 0){
                x = x+1;
                y = y-1;
            }
            else if(x_diff == 0 && y_diff < 0){
                y = y+1;
            }
            else if(x_diff == 0 && y_diff > 0){
                y = y-1;
            }
            else if(x_diff > 0 && y_diff > 0){
                x = x-1;
                y = y-1;
            }
            else if(x_diff > 0 && y_diff == 0){
                x = x-1;
            }
            else if(x_diff > 0 && y_diff < 0){
                x = x-1;
                y = y+1;
            }
            

            coordinate *m_cor = new coordinate;
            m_cor->x = x;
            m_cor->y = y;
            response->mr_content.move_to = *m_cor;
            //std::cerr <<"mr_content.move_to.x: " << response->mr_content.move_to.x<< " mr_content.move_to.y: " << response->mr_content.move_to.y << std::endl;
            response->mr_type=mr_move;
        }
        write(1,response,sizeof(monster_response));
        turn++;
    }
    exit(0);
    return 0;
}