
import math

import numpy as np
import torch
import psutil
from Model import *
from DatasetBuilder import *
import torch.multiprocessing as mp
from utils import *
import torch.distributions as td
import gc
# -------------------------------------------------------------------------------------------------------
def Worker(GlobalNet, DecisionPolicyOptimizer, DecisionValueOptimizer, DecisionAuxiliaryOptimizer, adder, Trainers_Number, crypto_name, amount=0, mediate=0,
           leverage=20, load_model=False):
    # -------------------------------------------------------------------------------------------------------

    LearningNet = TradingModel().float().cuda()
    LearningNet.train()

    LearningNet.BuildMemory()
    if load_model:
        LearningNet.load_state_dict(torch.load('./BaseModel/BaseModel' + crypto_name + '3.pt'))
        GlobalNet.load_state_dict(torch.load('./BaseModel/BaseModel' + crypto_name + '3.pt'))
    # -------------------------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------------------------
    count, Bankrupt = 0, 0
    decsion_count, trade_count, win_count = 0, 1e-6, 0
    profit, profit_count = 0, 1e-6
    tick = 1

    DelayCount=0
    # -------------------------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------------------------

    lossPSum,lossDSum=0,0
    lossPcount,lossDcount = 1e-5,1e-5
    virtual_trader = RL_Agent(leverage=leverage, testmode=True)
    testing_number_data = TakeCsvData('G:/CsvStorage/' + crypto_name + '/' + crypto_name + '_5M.csv')

    keepingPb_Save,V_tSave,DeicideAuxSave = [],[],[]
    # -------------------------------------------------------------------------------------------------------
    DecisionCount, PositionCount =0,0
    if amount == 0:
        episodes = len(testing_number_data) - adder - LeastNumber2Build - 57399
    else:
        episodes = amount - mediate

    for number in range(episodes):
        try:
            decided_number = number + adder + mediate
            url = r'G:/NumpyDataStorage/' + crypto_name + '/' + str(decided_number + 1) + '.npy'
            count += 1
            crypto_chart = np.load(url)

            # -------------------------------------------------------------------------------------------------------------
            video = torch.from_numpy(crypto_chart).float().reshape(-1, 1, 270, 240).cuda()

            close_price_in_csv_data = testing_number_data['ClosePrice'][
                decided_number + 57399 + LeastNumber2Build]

            # <------------------------------------------------------------------------------------------------------->
            if DelayCount==0:
                DelayCount =0
                if virtual_trader.CheckActionSelectMode == CheckActionSelectModeTrue:

                    K_exist = 0
                    LearningNet.BuildMemory()
                    with torch.no_grad():
                        np.save(r'G:/TrainingDatas/Position/' + str(Trainers_Number) + '_' + str(
                            PositionCount) + '.npy', np.load(url))
                        PositionCount += 1
                        Position_A_PRE = LearningNet.PositionAction(video, memory_mode=True)

                        CurrentPosition = virtual_trader.check_position(td.Categorical(Position_A_PRE).sample())

                    if CurrentPosition != POSITION_HOLD:
                        trade_count += 1
                    virtual_trader.PositionPrice = close_price_in_csv_data
                # <------------------------------------------------------------------------------------------------------->

                virtual_trader.checking_bankrupt()
                virtual_trader.check_price_type(close_price_in_csv_data)
                # <------------------------------------------------------------------------------------------------------->
                if virtual_trader.CheckActionSelectMode == CheckActionSelectModeFalse:
                    with torch.no_grad():
                        K_exist += 1
                        keeping, DecideAux = LearningNet.DecideAction(video, memory_mode=True)
                        K_V = LearningNet.DecideValue(video)
                        DeicideAuxSave.append(DecideAux)
                        keeping = td.Categorical(keeping)

                        np.save(r'G:/TrainingDatas/Decision/' + str(Trainers_Number) + '_' + str(
                            DecisionCount) + '.npy', np.load(url))
                        DecisionCount += 1
                        keepingPb_Save.append(keeping)
                        CheckKeep = keeping.sample()
                        virtual_trader.check_keeping(CheckKeep)
                        rewardD, DoesDone = virtual_trader.get_reward()
                        V_t = rewardD.cuda().detach() + 0.98 * K_V
                        V_tSave.append(V_t)

                # <------------------------------------------------------------------------------------------------------->
                if virtual_trader.CheckActionSelectMode == CheckActionSelectModeUNTRAINED:

                    LearningNet.BuildMemory()
                    if K_exist > 1:
                        learning_data = torch.from_numpy(np.load(
                            r'G:/TrainingDatas/Decision/' + str(Trainers_Number) + '_' + str(0) + '.npy')).reshape(
                            -1,1, 270, 240).float().cuda()
                        LearningNet.PositionAction(learning_data,memory_mode=True)
                        cnt=0
                        saved = None
                        for adv in reversed(V_tSave):
                            if cnt ==0:
                                V_tSave[len(V_tSave)-1-cnt]=adv
                                saved =adv
                            else:
                                V_tSave[len(V_tSave) - 1 - cnt] += 0.98*saved
                                saved = V_tSave[len(V_tSave) - 1 - cnt]
                            cnt+=1

                        for i in range(DecisionCount):
                            learning_data = torch.from_numpy(np.load(
                                r'G:/TrainingDatas/Decision/' + str(Trainers_Number) + '_' + str(i) + '.npy')).reshape(
                                -1,1, 270, 240).float().cuda()
                            PriorKeep, DecideAux = LearningNet.DecideAction(learning_data)
                            PriorKeep = td.Categorical(PriorKeep)

                            entropy = PriorKeep.entropy().mean()
                            new_log_prob = PriorKeep.log_prob(keepingPb_Save[i].sample())
                            log_ratio = new_log_prob - keepingPb_Save[i].log_prob(keepingPb_Save[i].sample())
                            ratio = torch.exp(log_ratio)
                            pg_losses = - V_t.detach() * ratio
                            pg_losses2 = -V_t * torch.clamp(ratio, 1.0 - 0.2, 1.0 + 0.2)
                            pg_losses = torch.max(pg_losses, pg_losses2).mean()

                            negativeEntropy = -entropy * 0.01

                            DecisionPiLoss = pg_losses + negativeEntropy + 0.01 * 0.5 * (log_ratio ** 2).mean()

                            DecisionPolicyOptimizer.zero_grad()
                            DecisionPiLoss.backward()
                            ensure_shared_grads(LearningNet, GlobalNet)
                            DecisionPolicyOptimizer.step()

                            V_K = LearningNet.DecideValue(learning_data)
                            DecisionValueLoss = 0.5 * ((V_K - V_t.detach()) ** 2).mean()

                            DecisionValueOptimizer.zero_grad()
                            DecisionValueLoss.backward()
                            ensure_shared_grads(LearningNet, GlobalNet)
                            DecisionValueOptimizer.step()
                            lossDSum += (DecisionValueLoss.item() + DecisionPiLoss.item())
                            lossDcount += 1
                            LearningNet.load_state_dict(GlobalNet.state_dict())

                        for i in range(DecisionCount):
                            learning_data = torch.from_numpy(np.load(
                                r'G:/TrainingDatas/Decision/' + str(Trainers_Number) + '_' + str(i) + '.npy')).reshape(
                                -1, 1, 270, 240).float().cuda()
                            oldpolicy = keepingPb_Save[i]
                            newpolicy, Auxilary = LearningNet.DecideAction(learning_data,memory_mode=True)

                            policy_distance = td.kl_divergence(oldpolicy, td.Categorical(newpolicy)).mean()
                            Value_loss = 0.5 * (
                                    (LearningNet.DecideValue(learning_data) - V_tSave[i].detach()) ** 2).mean()
                            DecideAuxLoss = 0.5 * ((Auxilary - DeicideAuxSave[i].detach()) ** 2).mean()
                            DecisionAuxiliaryOptimizer.zero_grad()
                            (policy_distance + Value_loss + DecideAuxLoss).backward()
                            ensure_shared_grads(LearningNet, GlobalNet)
                            DecisionAuxiliaryOptimizer.step()

                            LearningNet.load_state_dict(GlobalNet.state_dict())

                        DecisionCount = 0
                        keepingPb_Save.clear()
                        V_tSave.clear()
                        DeicideAuxSave.clear()

                    rewardP = virtual_trader.ChoicingTensor.cuda()

                    if rewardP.item() >= 0 and CurrentPosition != POSITION_HOLD:
                        win_count += 1
                    if rewardP.item() != 0 and CurrentPosition != POSITION_HOLD:
                        profit += rewardP.item()
                        profit_count += 1

                # <------------------------------------------------------------------------------------------------------->

                # <------------------------------------------------------------------------------------------------------->
                if DoesDone:
                    Bankrupt += 1


                if count > tick * 1000:
                    tick += 1
                    torch.save(GlobalNet.state_dict(), './BaseModel.pt')
                    torch.cuda.empty_cache()
                    print(f"{Trainers_Number}, "
                          f"{count + mediate}, "
                          f"{str(virtual_trader.PercentState)}  % Now,"
                          f"{str(virtual_trader.UndefinedPercent)}  % Profit,"
                          f" Bankrupt Count:  {str(Bankrupt)}"
                          f" Win Count:  {str(win_count / trade_count * 100)}"
                          f" Average Profit:  {str(profit / profit_count)}"
                          f" Trade Count:  {str(profit_count)}"
                          f" LossP mean:  {str(lossPSum / lossPcount)}"
                          f" LossD mean:  {str(lossDSum / lossDcount)}"
                          f" Storage Util:  {str(psutil.virtual_memory())}")
                gc.collect()

            else:
               DelayCount-=1

        # -------------------------------------------------------------------------------------------------------
        except KeyboardInterrupt:
            print("Finishing Training With Keyboard")
            torch.save(GlobalNet.state_dict(), './BaseModel/PreTrainedBaseModel' + crypto_name + '3.pt')
            return
        except np.core._exceptions._ArrayMemoryError:
            pass


       #  -------------------------------------------------------------------------------------------------------'''
    DrawingParameters(GlobalNet, crypto_name)
    torch.save(GlobalNet.state_dict(), './BaseModel/BaseModel' + crypto_name + '3.pt')


# -------------------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------------------
def TrainWithDataset(crypto_name, load_model=False, leverage=20, setting_stepper=50000, mediate=0, iter=5,
                     starter=20000):
    GlobalNet = TradingModel().float().cuda()  # global network
    GlobalNet.share_memory()

    DecisionPolicyOptimizer = GlobalAdamW( list(GlobalNet.Decisioner.parameters()) + list(GlobalNet.MemoryInterpreter.parameters()) + list(GlobalNet.DecisionA.parameters()), lr=1e-3)
    DecisionValueOptimizer = GlobalAdamW(list(GlobalNet.Decisioner.parameters()) + list(GlobalNet.MemoryInterpreter.parameters()) + list(GlobalNet.DecisionV.parameters()), lr=1e-3)
    DecisionAuxiliaryOptimizer = GlobalAdamW(list(GlobalNet.Decisioner.parameters()) + list(GlobalNet.MemoryInterpreter.parameters()) + list(GlobalNet.DecisionA.parameters()) + list(GlobalNet.DecisionAuxilary.parameters()) + list(GlobalNet.DecisionV.parameters()), lr=1e-3)
    workers = [mp.Process(target=Worker, args=(
        GlobalNet,
        DecisionPolicyOptimizer,DecisionValueOptimizer,DecisionAuxiliaryOptimizer,
        starter + (i) * setting_stepper, i + 1, crypto_name,
        setting_stepper, mediate, leverage, load_model)) for i in range(iter)]

    [w.start() for w in workers]
    [w.join() for w in workers]


# -------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    model = TradingModel()
    TrainStep = 1

    parameter_size(model)

    normalize_parameters(model)

    initialize_parameters(model)


    for i in range(TrainStep):
        TrainWithDataset('BTCUSDT', load_model=False, leverage=20, setting_stepper=int(330000/5), mediate=0, iter=5,
                         starter=30000)

  #  model.load_state_dict(torch.load('./BaseModel/BaseModelBTCUSDT3.pt'))
   # torch.save(model.state_dict(), './BaseModel/BaseModelSOLUSDT3.pt')
   # torch.save(model.state_dict(), './BaseModel/BaseModelETHUSDT3.pt')
   # torch.save(model.state_dict(), './BaseModel/BaseModelXRPUSDT3.pt')
   
    for i in range(TrainStep):
        TrainWithDataset('ETHUSDT', load_model=False, leverage=15, setting_stepper=int(330000/5), mediate=0, iter=5,
                         starter=30000)
    model.load_state_dict(torch.load('./BaseModel/BaseModelETHUSDT3.pt'))

    torch.save(model.state_dict(), './BaseModel/BaseModelSOLUSDT3.pt')
    torch.save(model.state_dict(), './BaseModel/BaseModelBTCUSDT3.pt')
    torch.save(model.state_dict(), './BaseModel/BaseModelXRPUSDT3.pt')
    for i in range(TrainStep):
        TrainWithDataset('SOLUSDT', load_model=True, leverage=10, setting_stepper=80000, mediate=0, iter=3,
                         starter=0)
    model.load_state_dict(torch.load('./BaseModel/BaseModelSOLUSDT3.pt'))

    torch.save(model.state_dict(), './BaseModel/BaseModelETHUSDT3.pt')
    torch.save(model.state_dict(), './BaseModel/BaseModelBTCUSDT3.pt')
    torch.save(model.state_dict(), './BaseModel/BaseModelXRPUSDT3.pt')
    for i in range(TrainStep):
        TrainWithDataset('XRPUSDT', load_model=True, leverage=7, setting_stepper=100000, mediate=0, iter=3,
                         starter=10000)
    model.load_state_dict(torch.load('./BaseModel/BaseModelXRPUSDT3.pt'))

    torch.save(model.state_dict(), './BaseModel/BaseModelETHUSDT3.pt')
    torch.save(model.state_dict(), './BaseModel/BaseModelBTCUSDT3.pt')
    torch.save(model.state_dict(), './BaseModel/BaseModelSOLUSDT3.pt')



