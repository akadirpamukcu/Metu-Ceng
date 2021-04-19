#include "logging.h"
#include "message.h"
#include "monster_class.hpp"
#include "monster_class.cpp"
#include "logging.c"

#include <sys/wait.h> 
#include <sys/types.h>
#include <unistd.h>
#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include<sys/socket.h>

#define PIPE(fd) socketpair(AF_UNIX, SOCK_STREAM, PF_UNIX, fd)

void monsterSort(std::vector<monster_class*>& monsters){
    bool swapped = true;
    int size = monsters.size();
    int rhs, lhs;
    monster_class* tmp = NULL;
    for (int i = 0; i < size; ++i){
        monsters[i]->xy= stoi(monsters[i]->x)*10 + stoi(monsters[i]->y);
    }
    for (int i = 0; i < size; ++i){
        for (int j = i+1; j < size; ++j){
        rhs = monsters[i]->xy;
        lhs = monsters[i]->xy;
        if (rhs > lhs){
            tmp = monsters[j];
            monsters[j] = monsters[i];
            monsters[i] = tmp;
            swapped = false;
        }
    }
        if (swapped){
            break;
        }

    }
}

bool is_move_valid(int type, coordinate target,std::string width , std::string height, std::string door_x,std::string door_y,std::vector<monster_class*>& monsters){
    if(target.x >= stoi(width) ) { return false;}
    if(target.y >= stoi(height) ) { return false;}
    if(type ==0){
        for(int i=0; i<monsters.size(); i++){
            if(target.x == stoi(monsters[i]->x) && target.y == stoi(monsters[i]->y)){return false;}
        }
    }
    if(type==1){
        if( target.x == stoi(door_x) && target.y == stoi(door_y) )  { return false;}
        if(target.x >= stoi(width) ) { return false;}
        if(target.y >= stoi(height) ) { return false;}
        for(int i=0; i<monsters.size(); i++){
            if(target.x == stoi(monsters[i]->x) && target.y == stoi(monsters[i]->y)){
                if(!(monsters[i]->is_dead))  {return false;}
            }
        }
    }
    return true;
}


            

void takeMonsters(int count, std::vector<monster_class*>& monsters ){
    for(int i=0; i<count; i++){
        std::string name;
        char symbol;
        std::string p_x,p_y;
        std::string arg1,arg2,arg3,arg4;
        std::cin >> name >> symbol >> p_x >> p_y >> arg1 >> arg2 >> arg3 >> arg4;
        std::vector<std::string> argv{name,arg1,arg2,arg3,arg4};
        monster_class* mon = new monster_class(name,symbol, p_x, p_y, argv);
        monsters[i] = mon;
    }
}

void  monster_coords(coordinate* monster_cor, int count, std::vector<monster_class*>& monsters){
    for(int i=0; i<count; i++){
        coordinate *new_cor = new coordinate;
        new_cor->x =  stoi(monsters[i]->x);
        new_cor->y = stoi(monsters[i]->y);
        monster_cor[i] = *new_cor;
    }
}

void forkMonsters(int count, std::vector<monster_class*> monsters,int** pipes, char** monster_forkargv){
    for(int i=0; i<count; i++){
        monsters[i]->pipe = new int[2];
        if (PIPE(monsters[i]->pipe) < 0) {
            std::cout << "monster PIPE error" << std::endl;
        }
        pipes[i]=monsters[i]->pipe;
    }
    for(int i=0; i<count; i++){
        if(!(monsters[i]->pid = fork())){
            //monster
            //player
            close(monsters[i]->pipe[0]);
            dup2(monsters[i]->pipe[1],0);
            dup2(monsters[i]->pipe[1],1);
            close(monsters[i]->pipe[1]);
            monster_forkargv[0] = new char[(monsters[i]->name).length() + 1];
            strcpy(monster_forkargv[0], (monsters[i]->name).c_str());
            for (int j = 1; j < 5; j++) {
                monster_forkargv[j] = new char[((monsters[i])->argv[j]).length() + 1];
                strcpy(monster_forkargv[j], ((monsters[i])->argv[j]).c_str());
                //std::cerr<< "arg" << j <<  " = " <<  monster_forkargv[j] << std::endl;
            }
            monster_forkargv[5] = (char*)NULL;
            if(execv(monster_forkargv[0], monster_forkargv)){
                printf("warning: execve returned an error.\n"); 
                exit(-1);
            }
        }
    }
}

int main(int argc, char **argv) {
    //std::cout << "start " << std::endl;
    player_response* p_response = new player_response;
    monster_response* m_response = new monster_response;
    monster_message* m_message = new monster_message;
    player_message* p_message = new player_message;
    m_response->mr_type= mr_dead;
    p_response->pr_type = pr_dead;
    m_message->game_over=false;
    p_message->game_over=false;
    std::string width,height,door_x, door_y, player_x, player_y;
    std::string max_attack_count, range, turn_number;
    std::string monster_count;
    int player_pid;
    std::string player;
    std::cin >> width >> height;
    std::cin >> door_x >> door_y;
    std::cin >> player_x >> player_y;
    std::cin >> player >> max_attack_count >> range >> turn_number;
    std::cin >> monster_count;
    std::vector<monster_class*> monsters(stoi(monster_count));
    monster_class* mon = new monster_class();
    
    takeMonsters(stoi(monster_count), monsters);
    char** monster_forkargv = new char*[9];
    int *player_pipe = new int[2];
    char** forkargv = new char*[7];
    if( PIPE(player_pipe) <0){
        std::cout<<"Player pipe error" << std::endl;
    }
    forkargv[0] = new char[player.length()+1];
    strcpy(forkargv[0], player.c_str());
    forkargv[1] = new char[door_x.length()+1];
    strcpy(forkargv[1], door_x.c_str());
    forkargv[2] = new char[door_y.length()+1];
    strcpy(forkargv[2], door_y.c_str());
    forkargv[3] = new char[max_attack_count.length()+1];
    strcpy(forkargv[3], max_attack_count.c_str());
    forkargv[4] = new char[range.length()+1];
    strcpy(forkargv[4], range.c_str());
    forkargv[5] = new char[turn_number.length()+1];
    strcpy(forkargv[5], turn_number.c_str());
    forkargv[6] = NULL;
    player_pid=fork();
    if(player_pid == 0){
        //player
        close(player_pipe[0]);
        dup2(player_pipe[1],0);
        dup2(player_pipe[1],1);
        close(player_pipe[1]);
        //std::cout << player.c_str()<<door_x.c_str()<< door_y.c_str()<< max_attack_count.c_str()<< range.c_str()<< turn_number.c_str()<<(char *)0;
        
        if(execv(forkargv[0], forkargv)){
            printf("warning: execve returned an error.\n"); 
            exit(-1);
        }
        printf("Player process should never get here\n");
        exit(42);
        
    }
    read(player_pipe[0], p_response, sizeof(player_response));
    if(p_response->pr_type == pr_ready){
        //std::cout<< "player is ready." << std::endl;
    }
    int** pipes = new int*[stoi(monster_count)];
    forkMonsters(stoi(monster_count),  monsters, pipes,  monster_forkargv);
    monsterSort(monsters);
    for(int i=0; i<stoi(monster_count); i++){
        read(monsters[i]->pipe[0], m_response, sizeof(monster_response));
        if( m_response->mr_type == mr_ready){
            //std::cout<<"monster " << i << " is ready." << std::endl;
        }
        else{
            std::cerr<<"ERR: monster " << i << " is not ready." << std::endl;
            return -1;
        }
    }
    struct timeval delay;
    int turn =0;
    p_message->total_damage = 0;
    int i=0;
    map_info *map = new map_info;
    map->door.x=stoi(door_x);
    map->door.y=stoi(door_y);
    map->map_height=stoi(height);
    map->map_width=stoi(width);
    game_over_status *g_over = new game_over_status;
   
    while(true){
        monster_coords(p_message->monster_coordinates, monsters.size(), monsters);
        p_message->alive_monster_count = (monsters.size());
        coordinate *p_cor = new coordinate;
        p_cor->x = stoi(player_x);
        p_cor->y = stoi(player_y);
        p_message->new_position= *p_cor;
        write(player_pipe[0], p_message, sizeof(player_message));
        p_message->total_damage=0;
        fd_set readfds;
        FD_ZERO(&readfds);
        FD_SET(player_pipe[0], &readfds);
        delay.tv_sec = 500 / 1000;
        delay.tv_usec = (500 % 1000) * 1000;
        map->alive_monster_count = monsters.size();
        monster_coords(map->monster_coordinates, monsters.size(), monsters);
        for(int i=0; i<(map->alive_monster_count); i++){
            map->monster_types[i] = monsters[i]->symbol;
        }
        map->player= p_message->new_position;
        print_map(map);
        int ret = select(player_pipe[0] + 1, &readfds, NULL, NULL, &delay);
        if (ret <= 0) {
            *g_over = go_left;
            for(i=0; i<monsters.size(); i++){
                monsters[i]->m_message->game_over = true;
                write(monsters[i]->pipe[0], m_message, sizeof(monster_message));
            }
            int player_status;
            waitpid(player_pid, &player_status,WNOHANG);
            break;   
        }
        read(player_pipe[0], p_response, sizeof(player_response));

        for(int i=0; i<monsters.size(); i++){
            monsters[i]->m_message->damage=0;
            coordinate *m_cor = new coordinate;
            m_cor->x = stoi(monsters[i]->x);
            m_cor->y = stoi(monsters[i]->y);
            monsters[i]->m_message->new_position.x = stoi(monsters[i]->x); 
            monsters[i]->m_message->new_position.y = stoi(monsters[i]->y); 
            monsters[i]->m_message->player_coordinate.x= stoi(player_x);
            monsters[i]->m_message->player_coordinate.y= stoi(player_y);
        }
        if(p_response->pr_type == pr_attack){
            //std::cout << "player attacks"<< std::endl;
            int size = sizeof(p_response->pr_content.attacked)/sizeof(p_response->pr_content.attacked[0]);
            for(i=0; i<monsters.size(); i++){
                monsters[i]->m_message->damage = p_response->pr_content.attacked[i];
            }
        }
        else if(p_response->pr_type == pr_move){
            //std::cout << "player moves"<< std::endl;
            if(p_response->pr_content.move_to.x == stoi(door_x) && p_response->pr_content.move_to.y == stoi(door_y)){
                
                for(i=0; i<monsters.size(); i++){
                    monsters[i]->m_message->game_over = true;
                    write(monsters[i]->pipe[0], m_message, sizeof(monster_message));
                }
                p_message->game_over=true;
                *g_over = go_reached;
                p_message->new_position=p_response->pr_content.move_to;
                int player_status;
                waitpid(player_pid, &player_status,WNOHANG);
                //std::cout << "player reached the door"<< std::endl;
                break;
            }
            if(is_move_valid(0, p_response->pr_content.move_to, width ,height, door_x, door_y, monsters)){
                //std::cout << "player's move is valid to: "<< p_response->pr_content.move_to.x << "," <<p_response->pr_content.move_to.y  << std::endl;
                p_message->new_position=p_response->pr_content.move_to;
                player_x = std::to_string(p_response->pr_content.move_to.x);
                player_y = std::to_string(p_response->pr_content.move_to.y);
                for(int i=0; i<monsters.size(); i++){
                    monsters[i]->m_message->player_coordinate=p_response->pr_content.move_to;
                }
            }
        }
        else if(p_response->pr_type == pr_dead){
            //std::cout << "player dies"<< std::endl;
            for(int i=0; i<monsters.size(); i++){
                monsters[i]->m_message->game_over = true;
                write(monsters[i]->pipe[0], m_message, sizeof(monster_message));
            }
            *g_over = go_died;
            int player_status;
            waitpid(player_pid, &player_status,WNOHANG);
            //std::cout << "player dead breaking"<< std::endl;
            break;
        }
        else{
            int player_status;
            waitpid(player_pid, &player_status,WNOHANG);
            break;
        } 

        for(int i=0; i<monsters.size(); i++){
            write(monsters[i]->pipe[0], monsters[i]->m_message, sizeof(monster_message));
            //std::cout << "writing monsters with "<< monsters[i]->m_message->damage << " damage" << std::endl;
        }
        for(int i=0; i<monsters.size(); i++){
            fd_set readfds;
            FD_ZERO(&readfds);
            FD_SET(monsters[i]->pipe[0], &readfds);
            delay.tv_sec = 50 / 1000;
            delay.tv_usec = (50 % 1000) * 1000;
            int ret = select(monsters[i]->pipe[0] + 1, &readfds, NULL, NULL, &delay);
            if (ret <= 0) {
                //std::cout << "monster " << i << " is late" <<std::endl;
            }
            //std::cout << "read monster "<<  i << " with pid : " << monsters[i]->pid <<  " turn :" << turn<<  std::endl;
            read(monsters[i]->pipe[0], monsters[i]->m_response, sizeof(monster_response));
        }
        monsterSort(monsters);
                

        for(int i=0; i<monsters.size(); i++){
            if(monsters[i]->m_response->mr_type == mr_dead){
                //std::cout << "monster " << i << " dies"<< std::endl;
                monsters[i]->is_dead=true;
                //std::cout << "monster left : " << monsters.size() << std::endl;                
            }
            if(monsters[i]->m_response->mr_type == mr_attack){
                int attack = (monsters[i]->m_response)->mr_content.attack;
                //std::cout << "monster attacks with  :  " <<  attack <<std::endl;
                p_message->total_damage+=monsters[i]->m_response->mr_content.attack;
            }
            else if(monsters[i]->m_response->mr_type == mr_move){
                //std::cout << "monster move  " << std::endl;
                if (is_move_valid(1, monsters[i]->m_response->mr_content.move_to, width, height, door_x, door_y, monsters )){
                    //std::cout << "monster move is valid to : "<< monsters[i]->m_response->mr_content.move_to.x << "," <<  monsters[i]->m_response->mr_content.move_to.x<<std::endl;
                    monsters[i]->x = std::to_string(monsters[i]->m_response->mr_content.move_to.x);
                    monsters[i]->y = std::to_string(monsters[i]->m_response->mr_content.move_to.y);
                }
            }
        }
        for(int i=0; i<monsters.size(); i++){
            if(monsters[i]->is_dead){
                int monster_status;
                waitpid(monsters[i]->pid, &monster_status,WNOHANG);
                monsters.erase(monsters.begin()+i);
            }
        }
        if(monsters.size() <= 0){
            p_message->game_over = true;
            write(player_pipe[0], p_message, sizeof(player_message));
            //std::cout << "there are no monsters left player win message sent and leaving" << std::endl;
            map->alive_monster_count = monsters.size();
            *g_over = go_survived;
            int player_status;
            waitpid(player_pid, &player_status,WNOHANG);
            break;
        }
        turn++;
    }
    monster_coords(map->monster_coordinates, monsters.size(), monsters);
    for(int i=0; i<(map->alive_monster_count); i++){
        map->monster_types[i] = monsters[i]->symbol;
    }
    map->player= p_message->new_position;
    print_map(map);
    print_game_over(*g_over);
    //std::cout << "waiting chgilds to die" << std::endl;
    for(int i =0; i<monsters.size(); i++){
        int status;
        waitpid(monsters[i]->pid, &status, WNOHANG);
    }
    for(int i=0;i<stoi(monster_count); i++){
        close(pipes[i][0]);
        close(pipes[i][1]);
    }
    close(player_pipe[0]);
    close(player_pipe[1]);
    return 0;
}

