#ifndef HOMEWORK2_WRITEOUTPUT_H
#define HOMEWORK2_WRITEOUTPUT_H

#ifdef __cplusplus
extern "C" {
#endif


#include <stdio.h>
#include <pthread.h>
#include <sys/time.h>

typedef enum Action {
    SENDER_CREATED,                 //0
    SENDER_DEPOSITED,               //1
    SENDER_STOPPED,                 //2
    RECEIVER_CREATED,               //3
    RECEIVER_PICKUP,                //4
    RECEIVER_STOPPED,               //5
    DRONE_CREATED,                  //6
    DRONE_DEPOSITED,                //7
    DRONE_PICKUP,                   //8
    DRONE_GOING,                    //9
    DRONE_ARRIVED,                  //10
    DRONE_STOPPED,                  //11
    HUB_CREATED,                    //12
    HUB_STOPPED                     //13
} Action;

typedef struct PackageInfo {
    int sender_id;
    int sending_hub_id;
    int receiver_id;
    int receiving_hub_id;
} PackageInfo;

typedef struct SenderInfo {
    int id;
    int current_hub_id;
    int remaining_package_count;
    PackageInfo *packageInfo;
} SenderInfo;

typedef struct ReceiverInfo {
    int id;
    int current_hub_id;
    PackageInfo *packageInfo;
} ReceiverInfo;

typedef struct DroneInfo {
    int id;
    int current_hub_id;
    int current_range;
    PackageInfo *packageInfo;
    int next_hub_id;
} DroneInfo;

typedef struct HubInfo {
    int id;
} HubInfo;

void InitWriteOutput();

unsigned long long GetTimestamp();

void PrintThreadId();

void FillPacketInfo(PackageInfo *packageInfo, int sender_id, int sending_hub_id, int receiver_id, int receiving_hub_id);

void FillSenderInfo(SenderInfo *senderInfo, int id, int current_hub_id, int remaining_package_count,
                    PackageInfo *packageInfo);

void FillReceiverInfo(ReceiverInfo *receiverInfo, int id, int current_hub_id, PackageInfo *packageInfo);

void FillDroneInfo(DroneInfo *droneInfo, int id, int current_hub_id, int current_range, PackageInfo *packageInfo,
                   int next_hub_id);

void FillHubInfo(HubInfo *hubInfo, int id);

void
WriteOutput(SenderInfo *senderInfo, ReceiverInfo *receiverInfo, DroneInfo *droneInfo, HubInfo *hubInfo, Action action);

#ifdef __cplusplus
}
#endif

#endif //HOMEWORK2_WRITEOUTPUT_H
