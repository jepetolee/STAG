
import psutil
import torch.optim

from Model import *
from DatasetBuilder import *
from utils import *
import torch.distributions as td
import gc


# -------------------------------------------------------------------------------------------------------
def Worker(GlobalNet, DecisionOptimizer, adder,
           Trainers_Number, crypto_name, amount=0, mediate=0,
           leverage=20, load_model=False):
    # -------------------------------------------------------------------------------------------------------

    GlobalNet.BuildMemory()
    if load_model:
        GlobalNet.load_state_dict(torch.load('./BaseModel/BaseModel' + crypto_name + '3.pt'))
    # -------------------------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------------------------
    count, Bankrupt = 0, 0
    decsion_count, trade_count, win_count = 0, 1e-6, 0
    profit, profit_count = 0, 1e-6
    tick = 1

    DelayCount = 0
    # -------------------------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------------------------

    lossDSum = 0
    stackForTrain= 48
    lossPcount, lossDcount = 1e-5, 1e-5
    virtual_trader = RL_Agent(leverage=leverage, testmode=True)
    testing_number_data = TakeCsvData('G:/CsvStorage/' + crypto_name + '/' + crypto_name + '_5M.csv')

    keepingPb_Save, V_tSave = [], []
    # -------------------------------------------------------------------------------------------------------
    DecisionCount, PositionCount = 0, 0
    Delay=0
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
            if DelayCount == 0:
                DelayCount = 0
                if virtual_trader.CheckActionSelectMode == CheckActionSelectModeTrue:

                    GlobalNet.BuildMemory()
                    with torch.no_grad():
                        np.save(r'G:/TrainingDatas/Position/' + str(Trainers_Number) + '_' + str(
                            PositionCount) + '.npy', np.load(url))
                        PositionCount += 1
                        Position_A_PRE = GlobalNet.PositionAction(video, memory_mode=True)

                        CurrentPosition = virtual_trader.check_position( torch.argmax(Position_A_PRE,dim=1))
                    virtual_trader.PositionPrice = close_price_in_csv_data
                    if CurrentPosition != POSITION_HOLD:
                        trade_count += 1
                    else:
                        Delay = 24

                # <------------------------------------------------------------------------------------------------------->

                virtual_trader.checking_bankrupt()
                virtual_trader.check_price_type(close_price_in_csv_data)
                # <------------------------------------------------------------------------------------------------------->
                if virtual_trader.CheckActionSelectMode == CheckActionSelectModeFalse and Delay <= 0 and CurrentPosition == POSITION_HOLD:
                    virtual_trader.CheckActionSelectMode = CheckActionSelectModeTrue
                elif virtual_trader.CheckActionSelectMode == CheckActionSelectModeFalse and Delay<=0 and CurrentPosition != POSITION_HOLD:
                    with torch.no_grad():
                        keeping,K_V = GlobalNet.DecideAction(video, memory_mode=True)
                        stackForTrain -=1

                        np.save(r'G:/TrainingDatas/Decision/' + str(Trainers_Number) + '_' + str(
                            DecisionCount) + '.npy', np.load(url))
                        DecisionCount += 1
                        keepingPb_Save.append(td.Categorical(keeping))
                        CheckKeep = torch.argmax(keeping,dim=1)
                        if CheckKeep.item() == 1:
                            Delay = 3
                        virtual_trader.check_keeping(CheckKeep)
                        rewardD, DoesDone = virtual_trader.get_reward()
                        if DoesDone:
                            Bankrupt += 1
                        V_t = rewardD.cuda().detach() + 0.98 * K_V
                        V_tSave.append(V_t)

                else:
                    Delay-=1
                # <------------------------------------------------------------------------------------------------------->
                if virtual_trader.CheckActionSelectMode == CheckActionSelectModeUNTRAINED or stackForTrain==0:
                    stackForTrain=48
                    Delay=0
                    virtual_trader.CheckActionSelectMode = CheckActionSelectModeTrue
                    learning_data = torch.from_numpy(np.load(
                        r'G:/TrainingDatas/Decision/' + str(Trainers_Number) + '_' + str(0) + '.npy')).reshape(
                        -1, 1, 270, 240).float().cuda()
                    GlobalNet.PositionAction(learning_data, memory_mode=True)
                    cnt = 0
                    saved = None
                    for adv in reversed(V_tSave):
                        if cnt == 0:
                            V_tSave[len(V_tSave) - 1 - cnt] = adv
                            saved = adv
                        else:
                            V_tSave[len(V_tSave) - 1 - cnt] += 0.9 * saved
                            saved = V_tSave[len(V_tSave) - 1 - cnt]
                        cnt += 1

                    GlobalNet.BuildMemory()
                    for i in range(DecisionCount):
                        ####################################################################################
                        learning_data = torch.from_numpy(np.load(
                            r'G:/TrainingDatas/Decision/' + str(Trainers_Number) + '_' + str(i) + '.npy')).reshape(
                            -1, 1, 270, 240).float().cuda()
                        PriorKeep,_ = GlobalNet.DecideAction(learning_data)

                        action_dist = td.Categorical(PriorKeep)
                        entropy = action_dist.entropy().mean()

                        ratio = torch.exp(action_dist.log_prob(keepingPb_Save[i].sample().detach()) - keepingPb_Save[i].log_prob(keepingPb_Save[i].sample().detach()))
                        surr1 = ratio * V_tSave[i]
                        surr2 = torch.clamp(ratio, 1 - 0.2, 1 + 0.2) * V_tSave[i]
                        policy_loss = -torch.min(surr1, surr2).mean()

                        DecisionOptimizer.zero_grad()
                        loss = policy_loss + 0.5 * entropy
                        loss.backward()
                        DecisionOptimizer.step()

                        lossDSum += loss.item()
                        lossDcount += 1
                    DecisionCount = 0
                    keepingPb_Save.clear()
                    V_tSave.clear()
                    rewardP = virtual_trader.ChoicingTensor.cuda()
                    if rewardP.item() >= 0 and CurrentPosition != POSITION_HOLD:
                        win_count += 1
                    if rewardP.item() != 0 and CurrentPosition != POSITION_HOLD:
                        profit += rewardP.item()
                        profit_count += 1



                # <------------------------------------------------------------------------------------------------------->

                if count > tick * 500:
                    tick += 1
                    torch.save(GlobalNet.state_dict(), './BaseModel/BaseModel' + crypto_name + '3.pt')
                    torch.cuda.empty_cache()
                    print(f"{Trainers_Number}, "
                          f"{count + mediate}, "
                          f"{str(virtual_trader.PercentState)}  % Now,"
                          f"{str(virtual_trader.UndefinedPercent)}  % Profit,"
                          f" Bankrupt Count:  {str(Bankrupt)}"
                          f" Win Count:  {str(win_count / trade_count * 100)}"
                          f" Average Profit:  {str(profit / profit_count)}"
                          f" Trade Count:  {str(profit_count)}"
                          f" LossD mean:  {str(lossDSum / lossDcount)}"
                          f" Storage Util:  {str(psutil.virtual_memory())}")
                gc.collect()

            else:
                DelayCount -= 1

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
def TrainWithDataset(crypto_name, load_model=False, leverage=20, setting_stepper=50000, mediate=0, starter=20000):
    GlobalNet = TradingModel().float().cuda()  # global network

    order=1
    DecisionOptimizer = torch.optim.AdamW(
        list(GlobalNet.Decisioner.parameters()) + list(GlobalNet.MemoryInterpreter.parameters()) + list(
            GlobalNet.DecisionA.parameters())+ list(
            GlobalNet.DecisionV.parameters()), lr=1e-3)
    Worker(GlobalNet,DecisionOptimizer, starter ,order, crypto_name, setting_stepper, mediate, leverage, load_model)


# -------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    model = TradingModel()
    TrainStep = 1

    parameter_size(model)

    normalize_parameters(model)

    initialize_parameters(model)

    for i in range(TrainStep):
        TrainWithDataset('BTCUSDT', load_model=False, leverage=20, setting_stepper=280000, mediate=0, starter=50000)

    model.load_state_dict(torch.load('./BaseModel/BaseModelBTCUSDT3.pt'))
    torch.save(model.state_dict(), './BaseModel/BaseModelSOLUSDT3.pt')
    torch.save(model.state_dict(), './BaseModel/BaseModelETHUSDT3.pt')
    torch.save(model.state_dict(), './BaseModel/BaseModelXRPUSDT3.pt')

    for i in range(TrainStep):
        TrainWithDataset('ETHUSDT', load_model=True, leverage=15, setting_stepper=280000, mediate=0, starter=50000)
    model.load_state_dict(torch.load('./BaseModel/BaseModelETHUSDT3.pt'))

    torch.save(model.state_dict(), './BaseModel/BaseModelSOLUSDT3.pt')
    torch.save(model.state_dict(), './BaseModel/BaseModelBTCUSDT3.pt')
    torch.save(model.state_dict(), './BaseModel/BaseModelXRPUSDT3.pt')
    for i in range(TrainStep):
        TrainWithDataset('SOLUSDT', load_model=True, leverage=10, setting_stepper=280000, mediate=0, starter=50000)
    model.load_state_dict(torch.load('./BaseModel/BaseModelSOLUSDT3.pt'))

    torch.save(model.state_dict(), './BaseModel/BaseModelETHUSDT3.pt')
    torch.save(model.state_dict(), './BaseModel/BaseModelBTCUSDT3.pt')
    torch.save(model.state_dict(), './BaseModel/BaseModelXRPUSDT3.pt')
    for i in range(TrainStep):
        TrainWithDataset('XRPUSDT', load_model=True, leverage=7, setting_stepper=280000, mediate=0, starter=100000)
    model.load_state_dict(torch.load('./BaseModel/BaseModelXRPUSDT3.pt'))

    torch.save(model.state_dict(), './BaseModel/BaseModelETHUSDT3.pt')
    torch.save(model.state_dict(), './BaseModel/BaseModelBTCUSDT3.pt')
    torch.save(model.state_dict(), './BaseModel/BaseModelSOLUSDT3.pt')



