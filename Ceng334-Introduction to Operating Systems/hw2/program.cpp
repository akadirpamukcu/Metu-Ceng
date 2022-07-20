#include <iostream>
#include <thread>
#include <map>
#include <vector>
#include <string>
#include <mutex>
#include <condition_variable>

#include "writeOutput.c"
#include "helper.c"

#include <unistd.h>

using namespace std;

bool TERMINATE = false;

int N = 0, N2 = 0;

int RUNNING_RECEIVERS = 0;
mutex RUNNING_RECEIVERS_MUTEX;
int TOTAL_PACKAGES = 0;
mutex TOTAL_PACKAGES_MUTEX;
int RECEIVED_PACKAGES = 0;
mutex RECEIVED_PACKAGES_MUTEX;

class ConditionalLock
{
    private:
    bool group = false;
    bool condition = false;
    mutex mtx;
    unique_lock<mutex> lock;
    condition_variable variable;
    public:
    ConditionalLock(bool is_group){group = is_group;}
    ConditionalLock(){}
    void wait()
    {
        unique_lock<mutex> lock(mtx);
        variable.wait(lock, [this]{return condition;});
        condition = group;
    }
    void notify()
    {
        condition = true;
        if (group) variable.notify_all();
        else variable.notify_one();
    }
};

class Sender
{
    public:

    SenderInfo info;

    ConditionalLock* lock;

    int id, speed, hub_id, total_package;

    Sender(){}
    Sender(int Id, int Speed, int Hub_id, int Total_package)
    {
        lock = new ConditionalLock();

        id = Id;
        speed = Speed;
        hub_id = Hub_id - 1;
        total_package = Total_package;

        FillSenderInfo(&info, id+1, hub_id+1, total_package, NULL);
    }
    int get_rand_id()
    {
        return rand() % N;
    }
    void start();
};

class Receiver
{
    public:

    ReceiverInfo info;

    bool terminate = false;
    ConditionalLock* lock;

    int id, speed, hub_id;

    Receiver(){}
    Receiver(int Id, int Speed, int Hub_id)
    {
        lock = new ConditionalLock();

        id = Id;
        speed = Speed;
        hub_id = Hub_id - 1;

        FillReceiverInfo(&info, id+1, hub_id+1, NULL);
    }
    void start();
};

class Drone
{
    public:

    DroneInfo info;

    bool terminate = false;
    ConditionalLock * lock;
    bool is_working = false;
    int caller_hub;
    long long int old_time;

    int id, speed, hub_id, max_range;

    int current_charge;

    Drone(){}
    Drone(int Id, int Speed, int Hub_id, int Max_range)
    {
        lock = new ConditionalLock();
        old_time = timeInMilliseconds();
        caller_hub = -1;

        id = Id;
        speed = Speed;
        hub_id = Hub_id - 1;
        max_range = Max_range;

        current_charge = max_range;

        FillDroneInfo(&info, id+1, hub_id+1, max_range, NULL, 0);
    }
    void charge(int required) {
        if (current_charge >= required) {
            return;
        }
        else {
            // printf("Drone %d charging...\n", id);
            long long int current_time = timeInMilliseconds();
            current_charge = calculate_drone_charge(current_time-old_time, current_charge, max_range);
            return charge(required);
        }
    }
    void start();
};

class Hub
{
    public:

    HubInfo info;

    bool terminate = false;
    ConditionalLock * lock;
    
    int id, incoming_space, outgoing_space, charging_space;

    int incoming_loaded = 0, outgoing_loaded = 0, charging_loaded = 0;

    int* distances;

    vector<PackageInfo> incoming;
    int inSize = 0;
    vector<PackageInfo> outgoing;

    Hub(){}
    Hub(int Id, int Incoming_space, int Outgoing_space, int Charging_space, int* dists)
    {
        lock = new ConditionalLock();

        id = Id;
        incoming_space = Incoming_space;
        outgoing_space = Outgoing_space;
        charging_space = Charging_space;

        distances = new int[N];
        copy(dists, dists+N, distances);

        FillHubInfo(&info, id+1);
    }
    bool can_deposit() {
        while (outgoing.size() == outgoing_space) {
            if (TERMINATE) return false;
        }
        return true;
    }
    int find_drone(Drone* drones) {
        int call_drone_id = -1;
        int invite_drone_id = -1;
        int range = 0;
        for(int i = 0; i < N2; i++) {
            if (!drones[i].is_working) {
                if (drones[i].hub_id == id) {
                    if (call_drone_id == -1) {
                        call_drone_id = i;
                        range = drones[i].current_charge;
                    }
                    else if (drones[i].current_charge > range) {
                        call_drone_id = i;
                        range = drones[i].current_charge;
                    }
                }
                else if (call_drone_id == -1) {
                    if (range == 0) {
                        range = drones[i].current_charge;
                        invite_drone_id = i;
                    }
                    else if (drones[i].current_charge > range) {
                        range = drones[i].current_charge;
                        invite_drone_id = i;
                    }
                }
            }
        }
        if (call_drone_id != -1) {
            drones[call_drone_id].is_working = true;
            return call_drone_id;
        }
        else if (invite_drone_id != -1) {
            drones[invite_drone_id].is_working = true;
        }
        return invite_drone_id;
    }
    void start();
};

ConditionalLock starter(true);
Hub* hubs;
Sender* senders;
Receiver* receivers;
Drone* drones;

void Sender::start()
{
    TOTAL_PACKAGES_MUTEX.lock();
    TOTAL_PACKAGES += total_package;
    TOTAL_PACKAGES_MUTEX.unlock();
    WriteOutput(&info, NULL, NULL, NULL, SENDER_CREATED);
    // printf("Sender id: %d, speed: %d, hub_id: %d, packages: %d\n",
    //         id, speed, hub_id, total_package);
    starter.wait();
    // printf("Sender Released %d\n", id);
        while (total_package > 0) {
            int recv_id = -1;
            while (1){
                recv_id = get_rand_id();
                if (recv_id==id){
                    continue;
                }
            break;
        }
        PackageInfo p;
        p.sender_id = id;
        p.sending_hub_id = hub_id;
        p.receiver_id = recv_id;
        p.receiving_hub_id = receivers[recv_id].hub_id;
        // printf("Sender %d package ready sender: %d, s_hub: %d, receiver: %d, r_hub: %d\n",
        //         id, p.sender_id, p.sending_hub_id, p.receiver_id, p.receiving_hub_id);
        if (hubs[hub_id].can_deposit()) {
            hubs[hub_id].outgoing.push_back(p);
            // printf("Sender %d deposited\n", id);
            hubs[hub_id].lock->notify();
            total_package--;
        }
        p.sender_id++;
        p.sending_hub_id++;
        p.receiver_id++;
        p.receiving_hub_id++;
        FillSenderInfo(&info, info.id, hub_id+1, total_package, &p);
        WriteOutput(&info, NULL, NULL, NULL, SENDER_DEPOSITED);
        p.sender_id--;
        p.sending_hub_id--;
        p.receiver_id--;
        p.receiving_hub_id--;
        wait(UNIT_TIME*speed);
    }
    FillSenderInfo(&info, info.id, hub_id+1, total_package, NULL);
    WriteOutput(&info, NULL, NULL, NULL, SENDER_STOPPED);
    // printf("\t\tSender %d terminating\n", id);
}

void Receiver::start()
{
    // printf("Receiver id: %d, speed: %d, hub_id: %d\n",
    //         id, speed, hub_id);
    starter.wait();
    // printf("Receiver Released %d\n", id);
    WriteOutput(NULL, &info, NULL, NULL, RECEIVER_CREATED);
    while (true) {
        // printf("Receiver %d is waiting...\n", id);
        // lock->wait();
        if (terminate || TERMINATE) {
            // printf("\t\tReceiver %d terminating\n", id);
            break;
        }
        // printf("Receiver %d got signal\n", id);
        if (hubs[hub_id].inSize > 0) {
            for (int i = 0; i < hubs[hub_id].inSize; i++) {
                if (hubs[hub_id].incoming[i].receiver_id == id) {
                    PackageInfo& P = hubs[hub_id].incoming[i];
                    PackageInfo p;
                    p.sender_id = P.sender_id+1;
                    p.sending_hub_id = P.sending_hub_id+1;
                    p.receiver_id = P.receiver_id+1;
                    p.receiving_hub_id = P.receiving_hub_id+1;
                    FillReceiverInfo(&info, info.id, hub_id+1, &p);
                    WriteOutput(NULL, &info, NULL, NULL, RECEIVER_PICKUP);
                    // printf("Receiver %d got a package sender %d, s_hub: %d\t\tTotal: %d, Remain: %d\n",
                    //         id, p.sender_id, p.sending_hub_id, TOTAL_PACKAGES, RECEIVED_PACKAGES+1);
                    hubs[hub_id].inSize--;
                    hubs[hub_id].incoming.erase(hubs[hub_id].incoming.begin()+i);
                    RECEIVED_PACKAGES_MUTEX.lock();
                    RECEIVED_PACKAGES++;
                    RECEIVED_PACKAGES_MUTEX.unlock();
                    break;
                }
                else {
                    // printf("Receiver %d found an unknown package\n", id);
                }
            }
        }
        // else {
        //     printf("Receiver %d no packages\n", id);
        // }
        if (RECEIVED_PACKAGES >= TOTAL_PACKAGES) {
            hubs[hub_id].terminate = true;
            hubs[hub_id].lock->notify();
            break;
        }
        wait(UNIT_TIME*speed);
    }
    RUNNING_RECEIVERS_MUTEX.lock();
    RUNNING_RECEIVERS--;
    RUNNING_RECEIVERS_MUTEX.unlock();
    for (int i = 0; i < N; i++) {
        receivers[i].lock->notify();
    }
    if (RUNNING_RECEIVERS < 1) {
        TERMINATE = true;
        hubs[hub_id].terminate = true;
        hubs[hub_id].lock->notify();
        // printf("\t\tReceiver %d terminating with sig\n", id);
        // return;
    }
    info.packageInfo = NULL;
    WriteOutput(NULL, &info, NULL, NULL, RECEIVER_STOPPED);
    // printf("\t\tReceiver %d terminating\n", id);
}

void Drone::start()
{
    // printf("Drone id: %d, speed: %d, hub_id: %d, range: %d\n",
    //         id, speed, hub_id, max_range);
    starter.wait();
    // printf("Drone Released %d\n", id);
    WriteOutput(NULL, NULL, &info, NULL, DRONE_CREATED);
    while (true) {
        // printf("Drone %d waiting...\n", id);
        lock->wait();
        if (terminate || TERMINATE) {
            // printf("\t\tDrone %d terminating\n", id);
            break;
        }
        // printf("Drone %d got signal\n", id);
        if (caller_hub == hub_id) {
            // printf("Drone %d got call\n", id);
            if (hubs[hub_id].outgoing.size() > 0) {
                PackageInfo& p = hubs[hub_id].outgoing[0];
                PackageInfo P;
                P.sender_id = p.sender_id;
                P.sending_hub_id = p.sending_hub_id;
                P.receiver_id = p.receiver_id;
                P.receiving_hub_id = p.receiving_hub_id;
                hubs[hub_id].outgoing.erase(hubs[hub_id].outgoing.begin());
                // printf("Drone %d taking package sender %d, s_hub: %d, receiver: %d, r_hub: %d\n",
                //         id, P.sender_id, P.sending_hub_id, P.receiver_id, P.receiving_hub_id);
                int range_required = range_decrease(hubs[hub_id].distances[P.receiving_hub_id], speed);
                charge(range_required);
                P.sender_id++;
                P.sending_hub_id++;
                P.receiver_id++;
                P.receiving_hub_id++;
                FillDroneInfo(&info, info.id, hub_id+1, current_charge, &P, P.receiving_hub_id);
                WriteOutput(NULL, NULL, &info, NULL, DRONE_PICKUP);
                P.sender_id--;
                P.sending_hub_id--;
                P.receiver_id--;
                P.receiving_hub_id--;
                travel(hubs[hub_id].distances[P.receiving_hub_id], speed);
                hubs[P.receiving_hub_id].incoming.push_back(P);
                hubs[P.receiving_hub_id].inSize++;
                hub_id = P.receiving_hub_id;
                // printf("Drone %d droping package sender %d, s_hub: %d, receiver: %d, r_hub: %d\n",
                //         id, P.sender_id, P.sending_hub_id, P.receiver_id, P.receiving_hub_id);
                P.sender_id++;
                P.sending_hub_id++;
                P.receiver_id++;
                P.receiving_hub_id++;
                FillDroneInfo(&info, info.id, hub_id+1, current_charge, &P, 0);
                WriteOutput(NULL, NULL, &info, NULL, DRONE_DEPOSITED);
                P.sender_id--;
                P.sending_hub_id--;
                P.receiver_id--;
                P.receiving_hub_id--;
                old_time = timeInMilliseconds();
                // receivers[P.receiver_id].lock->notify();
            }
            // else {
            //     printf("Drone %d call but empty packages\n", id);
            // }
        }
        else {
            // printf("Drone %d got invitation from %d\n", id, caller_hub);
            int range_required = range_decrease(hubs[hub_id].distances[caller_hub], speed);
            charge(range_required);
            FillDroneInfo(&info, info.id, hub_id+1, current_charge, NULL, caller_hub);
            WriteOutput(NULL, NULL, &info, NULL, DRONE_GOING);
            travel(hubs[hub_id].distances[caller_hub], speed);
            hub_id = caller_hub;
            FillDroneInfo(&info, info.id, hub_id+1, current_charge, NULL, 0);
            WriteOutput(NULL, NULL, &info, NULL, DRONE_ARRIVED);
            if (hubs[hub_id].outgoing.size() > 0) {
                PackageInfo& p = hubs[hub_id].outgoing[0];
                PackageInfo P;
                P.sender_id = p.sender_id;
                P.sending_hub_id = p.sending_hub_id;
                P.receiver_id = p.receiver_id;
                P.receiving_hub_id = p.receiving_hub_id;
                hubs[hub_id].outgoing.erase(hubs[hub_id].outgoing.begin());
                int range_required = range_decrease(hubs[hub_id].distances[P.receiving_hub_id], speed);
                old_time = timeInMilliseconds();
                charge(range_required);
                P.sender_id++;
                P.sending_hub_id++;
                P.receiver_id++;
                P.receiving_hub_id++;
                FillDroneInfo(&info, info.id, hub_id+1, current_charge, &P, P.receiving_hub_id);
                WriteOutput(NULL, NULL, &info, NULL, DRONE_PICKUP);
                P.sender_id--;
                P.sending_hub_id--;
                P.receiver_id--;
                P.receiving_hub_id--;
                travel(hubs[hub_id].distances[P.receiving_hub_id], speed);
                hubs[P.receiving_hub_id].incoming.push_back(P);
                hubs[P.receiving_hub_id].inSize++;
                // printf("Drone %d droping package sender %d, s_hub: %d, receiver: %d, r_hub: %d\n",
                //         id, P.sender_id, P.sending_hub_id, P.receiver_id, P.receiving_hub_id);
                P.sender_id++;
                P.sending_hub_id++;
                P.receiver_id++;
                P.receiving_hub_id++;
                FillDroneInfo(&info, info.id, hub_id+1, current_charge, &P, 0);
                WriteOutput(NULL, NULL, &info, NULL, DRONE_DEPOSITED);
                P.sender_id--;
                P.sending_hub_id--;
                P.receiver_id--;
                P.receiving_hub_id--;
                old_time = timeInMilliseconds();
                receivers[P.receiver_id].lock->notify();
            }
            // else {
            //     printf("Drone %d invited but empty packages\n", id);
            // }
        }
        is_working = false;
        // charge(max_range);
        FillDroneInfo(&info, info.id, hub_id+1, current_charge, NULL, 0);
        WriteOutput(NULL, NULL, &info, NULL, DRONE_STOPPED);
    }
}

void Hub::start()
{
    // printf("Hub id: %d, incoming_space: %d, outgoing_space: %d, charging_space: %d\n",
    //         id, incoming_space, outgoing_space, charging_space);
    starter.wait();
    // printf("Hub Released %d\n", id);
    WriteOutput(NULL, NULL, NULL, &info, HUB_CREATED);
    while(true) {
        // printf("Hub %d waiting...\n", id);
        lock->wait();
        if (terminate || TERMINATE) {
            // printf("\t\tHub %d terminating\n", id);
            for (int i = 0; i < N; i++) {
                if (receivers[i].hub_id == id) {
                    receivers[i].terminate = true;
                    receivers[i].lock->notify();
                    break;
                }
            }
            for (int i = 0; i < N2; i++) {
                drones[i].terminate = true;
                drones[i].lock->notify();
            }
            break;
        }
        // printf("Hub %d got signal\n", id);
        if (outgoing.size() > 0) {
            for (int i = 0; i < outgoing.size(); i++) {
                if (outgoing[i].receiving_hub_id == id) {
                    // printf("Hub %d found package for own receiver\n", id);
                    PackageInfo& p = outgoing[i];
                    PackageInfo P;
                    P.sender_id = p.sender_id;
                    P.sending_hub_id = p.sending_hub_id;
                    P.receiver_id = p.receiver_id;
                    P.receiving_hub_id = p.receiving_hub_id;
                    outgoing.erase(outgoing.begin()+i);
                    incoming.push_back(P);
                    inSize++;
                    for (int i = 0; i < N; i++) {
                        if (receivers[i].hub_id == id) {
                            receivers[i].lock->notify();
                            break;
                        }
                    }
                }
                else {
                    // printf("Hub %d finding drone\n", id);
                    int drone_id = find_drone(drones);
                    while(drone_id == -1 || drone_id >= N2) {
                        if (TERMINATE) break;
                        drone_id = find_drone(drones);
                    }
                    // printf("Hub %d found drone %d\n", id, drone_id);
                    drones[drone_id].caller_hub = id;
                    drones[drone_id].lock->notify();
                }
            }
            if (outgoing.size() > 0) lock->notify();
            if (inSize > 0) {
                for (int i = 0; i < N; i++) {
                    if (receivers[i].hub_id == id) {
                        receivers[i].lock->notify();
                        lock->notify();
                        break;
                    }
                }
            }
        }
    }
    WriteOutput(NULL, NULL, NULL, &info, HUB_STOPPED);
}

void thread_hub(int id)
{
    hubs[id].start();
}
void thread_sender(int id)
{
    senders[id].start();
}
void thread_receiver(int id)
{
    receivers[id].start();
}
void thread_drone(int id)
{
    drones[id].start();
}

int main(int argc, char** argv)
{
    cin>>N;

    hubs = new Hub[N];
    senders = new Sender[N];
    receivers = new Receiver[N];

    int dists[N];
    int input_size;
    int output_size;
    int drone_size;
    for(int i = 0; i < N; i++) {
        cin>>input_size;
        cin>>output_size;
        cin>>drone_size;
        for (int j = 0; j < N; j++) {
            cin >> dists[j];
        }
        hubs[i] = Hub(i, input_size, output_size, drone_size, dists);
    }
    
    int speed;
    int hub;
    int packages;
    for (int i = 0; i < N; i++) {
        cin>>speed;
        cin>>hub;
        cin>>packages;
        senders[i] = Sender(i, speed, hub, packages);
    }
    
    for (int i = 0; i < N; i++) {
        cin>>speed;
        cin>>hub;
        receivers[i] = Receiver(i, speed, hub);
    }

    cin>>N2;

    drones = new Drone[N2];

    int range;
    for (int i = 0; i < N2; i++) {
        cin>>speed;
        cin>>hub;
        cin>>range;
        drones[i] = Drone(i, speed, hub, range);
    }

    RUNNING_RECEIVERS = N;
    TOTAL_PACKAGES = 0;
    RECEIVED_PACKAGES = 0;
    
    srand(time(NULL));

    map<string, thread> threads;
    
    for (int i = 0; i < N; i++) {
        threads[string("h")+to_string(i)] = thread(thread_hub, i);
    }
    for (int i = 0; i < N; i++) {
        threads[string("s")+to_string(i)] = thread(thread_sender, i);
    }
    for (int i = 0; i < N; i++) {
        threads[string("r")+to_string(i)] = thread(thread_receiver, i);
    }
    for (int i = 0; i < N2; i++) {
        threads[string("d")+to_string(i)] = thread(thread_drone, i);
    }

    starter.notify();
    
    for (auto& t : threads) {
        t.second.join();
    }

    return 0;
}