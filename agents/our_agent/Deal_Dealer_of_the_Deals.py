import logging
from random import randint, random
from typing import cast, Dict, List, Set, Collection

from geniusweb.actions.Accept import Accept
from geniusweb.actions.Action import Action
from geniusweb.actions.Offer import Offer
from geniusweb.actions.PartyId import PartyId
from geniusweb.bidspace.AllBidsList import AllBidsList
from geniusweb.bidspace.BidsWithUtility import BidsWithUtility
from geniusweb.bidspace.Interval import Interval
from geniusweb.inform.ActionDone import ActionDone
from geniusweb.inform.Finished import Finished
from geniusweb.inform.Inform import Inform
from geniusweb.inform.OptIn import OptIn
from geniusweb.inform.Settings import Settings
from geniusweb.inform.YourTurn import YourTurn
from geniusweb.issuevalue.Bid import Bid
from geniusweb.party.Capabilities import Capabilities
from geniusweb.party.DefaultParty import DefaultParty
from geniusweb.profileconnection.ProfileConnectionFactory import (
    ProfileConnectionFactory,
)
from geniusweb.progress.ProgressRounds import ProgressRounds
from geniusweb.profileconnection.ProfileInterface import ProfileInterface
from geniusweb.profile.utilityspace.LinearAdditive import LinearAdditive
from geniusweb.progress.Progress import Progress
from tudelft.utilities.immutablelist.ImmutableList import ImmutableList
from time import sleep, time as clock
from decimal import Decimal
from tudelft_utilities_logging.Reporter import Reporter
from collections import defaultdict
import math
import numpy as np


class Deal_Dealer(DefaultParty):
    # value initializations
    def __init__(self, reporter: Reporter = None):
        super().__init__(reporter)
        self._profileint: ProfileInterface = None
        self._utilspace: LinearAdditive = None
        self._me: PartyId = None
        self._progress: Progress = None
        self._lastReceivedBid: Bid = None
        self._bestReceivedBid: Bid = None  # variable to store the best offer received from the opposition
        self._e: float = 0.8
        self._counter: int = 1
        self._threshold: float = 0.45
        self._settings: Settings = None
        self.getReporter().log(logging.INFO, "party is initialized")

        self._optionCounts = defaultdict(lambda: defaultdict(int))
        self._optionIndex = defaultdict(lambda: defaultdict(int))
        self._allBidsList = None  # To store all possible bids


    # Competition type (do not change this function)
    def getCapabilities(self) -> Capabilities:
        """
        Parties are given turns in a round-robin order. When a party has the turn,
        it can accept, offer, or end negotiation. A deal is reached if all parties,
        accept an offer.

        More info at https://tracinsy.ewi.tudelft.nl/pubtrac/GeniusWeb#SessionProtocol
        """
        return Capabilities(
            set(["SAOP"]),
            set(["geniusweb.profile.utilityspace.LinearAdditive"]),
        )

    def notifyChange(self, info: Inform):
        """
        This is the entry point of all interaction with your agent after is has been initialized.

        Args:
            info (Inform): Contains either a request for action or information.
            See more details at: https://tracinsy.ewi.tudelft.nl/pubtrac/GeniusWeb#Inform
        """
        try:
            # parse Inform object, based on its specified type
            if isinstance(info, Settings):
                # Enter setting information. This is called once at the beginning of a session.
                # You can add your own code based on your own implementation.
                self._settings = info
                self._me = self._settings.getID()
                self._progress = self._settings.getProgress()
                newe = self._settings.getParameters().get("e")
                if newe != None:
                    if isinstance(newe, float):
                        self._e = newe
                    else:
                        self.getReporter().log(
                            logging.WARNING,
                            "parameter e should be Double but found " + str(newe),
                        )
                self._profileint = ProfileConnectionFactory.create(self._settings.getProfile().getURI(), self.getReporter())  # TODO
                # Initializing all possible bid counts to 0
                self._allBidsList = AllBidsList(self._profileint.getProfile().getDomain())
                self._initializeOptionCounts()

            elif isinstance(info, ActionDone):
                otheract: Action = info.getAction()
                if isinstance(otheract, Offer):
                    self._lastReceivedBid = otheract.getBid()

            elif isinstance(info, YourTurn):
                # Updating the opponent preferences
                if self._lastReceivedBid is not None:
                    self._counter += 1
                    self._updateOptionCounts(self._lastReceivedBid)
                # Before our play we keep track of the opponents proposal
                if self._isBetter(self._lastReceivedBid, self._bestReceivedBid):
                    self._bestReceivedBid = self._lastReceivedBid
                    utility = self._utilspace.getUtility(self._bestReceivedBid) if self._utilspace else 0
                    # print(f"Best Received Bid: {self._bestReceivedBid}, Utility: {utility}")
                self._myTurn()

            elif isinstance(info, Finished):
                self.getReporter().log(logging.INFO, "Final outcome:" + str(info))
                # print("TIME TAKEN:", 10*self._progress.get(round(clock() * 1000)))
                # self._printOptionCounts()
                # print(self._evaluate_opponent())
                self.terminate()
                # stop this party and free resources.
        except Exception as ex:
            self.getReporter().log(logging.CRITICAL, "Failed to handle info", ex)
        self._updateRound(info)

    def _isBetter(self, newBid: Bid, currentBest: Bid) -> bool:
        """
        Used to keep track of the best offer, offered thus far
        Checking the new offer and comparing its utility to the best one so far
        """
        if newBid is None:
            return False
        if currentBest is None:
            return True
        return (self._utilspace.getUtility(newBid) if self._utilspace else 0) > (self._utilspace.getUtility(currentBest) if self._utilspace else 0)

    def _initializeOptionCounts(self):
        for bid in self._allBidsList:
            for issue, option in bid.getIssueValues().items():
                if issue not in self._optionCounts:
                    self._optionCounts[issue] = {}
                    self._optionIndex[issue] = {}
                if option not in self._optionCounts[issue]:
                    self._optionCounts[issue][option] = 0
                    self._optionIndex[issue][option] = 0

    def _updateOptionCounts(self, bid: Bid):
        """
        Updating the counts of each option in each issue based on the opponent's bid.
        """
        for issue, option in bid.getIssueValues().items():
            self._optionCounts[issue][option] += 1
            self._optionIndex[issue][option] = self._optionCounts[issue][option] / self._counter

    def _printOptionCounts(self):
        """
        Prints the count of each option for each issue based on the opponent's bids.
        """
        print("Option Counts:")
        for issue, options in self._optionCounts.items():
            print(f"Issue: {issue}")
            for option, count in options.items():
                print(f"  Option: {option}, Count: {count}, Index: {self._optionIndex[issue][option]}")

    def _updateE(self):
        """
        @return the E value that controls the party's behaviour. Depending on the
                value of e, extreme sets show clearly different patterns of
               behaviour [1]:

               1. Boulware: For this strategy e &lt; 1 and the initial offer is
                maintained till time is almost exhausted, when the agent concedes
                up to its reservation value.

                2. Conceder: For this strategy e &gt; 1 and the agent goes to its
                reservation value very quickly.

                3. When e = 1, the price is increased linearly.

                4. When e = 0, the agent plays hardball.
        """
        # we are updating the e value based on our evaluation of the opponent
        score = self._evaluate_opponent()
        if score > 0.3:
            self._e = 0.75
            self._threshold = 0.4
        else:
            self._e = 0.85
            self._threshold = 0.45

    # Override
    def getDescription(self) -> str:
        return (
                "Multiagent Systems 2023-2024 ANAC Project "
                + "team 2: "
                + "Koutounidis Christos-Angelos AM:2019030138 "
                + "Mavrikaki Filia              AM:2019030071 "
                + "Farmakis Konstantinos        AM:2019030102 "
                + " "
        )

    # Override
    def terminate(self):
        self.getReporter().log(logging.INFO, "party is terminating:")
        super().terminate()
        if self._profileint is not None:
            self._profileint.close()
            self._profileint = None

    ##################### private support funcs #########################

    def _updateRound(self, info: Inform):
        """
        Update {@link #progress}, depending on the last received {@link Inform}

        @param info the received info.
        """
        if self._settings == None:  # not yet initialized
            return

        if not isinstance(info, OptIn):
            return

        # if we get here, round must be increased.
        if isinstance(self._progress, ProgressRounds):
            self._progress = self._progress.advance()

    def _myTurn(self):
        self._updateUtilSpace()
        elapsedTime = 10*self._progress.get(round(clock() * 1000))
        myAction: Action

        if elapsedTime > 9.75 and self._bestReceivedBid is not None and self._utilspace.getUtility(self._bestReceivedBid) > self._threshold:
            # If more than 9.75 seconds have passed, offer the best received bid if it is an ok one
            # (self_utility > threshold)
            # Or if we get an offer higher or equal to our best received offer we accept
            if self._lastReceivedBid != None and self._utilspace.getUtility(self._lastReceivedBid) >= self._utilspace.getUtility(self._bestReceivedBid):
                myAction = Accept(self._me, self._lastReceivedBid)
            else:
                myAction = Offer(self._me, self._bestReceivedBid)

        else:
            bid = self._makeBid()
            if elapsedTime < 3:
                if bid == None or (self._lastReceivedBid != None
                                   and self._utilspace.getUtility(self._lastReceivedBid) >= 0.9
                ):
                    myAction = Accept(self._me, self._lastReceivedBid)
                else:
                    myAction = Offer(self._me, bid)
            else:
                if bid == None or (
                        self._lastReceivedBid != None
                        and self._utilspace.getUtility(self._lastReceivedBid) >= self._utilspace.getUtility(bid)
                ):
                    # if bid==null we failed to suggest next bid.
                    myAction = Accept(self._me, self._lastReceivedBid)
                else:
                    myAction = Offer(self._me, bid)

        self.getConnection().send(myAction)

    def _updateUtilSpace(self) -> LinearAdditive:  # throws IOException
        newutilspace = self._profileint.getProfile()
        if not newutilspace == self._utilspace:
            self._utilspace = cast(LinearAdditive, newutilspace)
        return self._utilspace

    def _makeBid(self) -> Bid:
        """
        @return next possible bid with current target utility, or null if no such
                bid.
        """
        time = 10*self._progress.get(round(clock() * 1000))

        self._bidutils = BidsWithUtility.create(self._utilspace)
        self._computeMinMax(time)

        options: ImmutableList[Bid] = self.getBids()
        if options.size() == 0:
            # if we can't find good bid, get max util bid....
            options = self.getBids()

        if time > 3:
            options2 = self._Fictitious_Play(options)
            return options2[randint(0, len(options2) - 1)]
        else:
            # pick a random one
            return options.get(randint(0, options.size() - 1))

    def getBids(self) -> ImmutableList[Bid]:
        """
        @param utilityGoal the requested utility
        @return bids with utility inside [utilitygoal-{@link #tolerance},
                utilitygoal]
        """
        return self._bidutils.getBids(
            Interval(self._minUtil, self._maxUtil)
        )

    def _Fictitious_Play(self, allBids: ImmutableList[Bid]) -> List[Bid]:
        """
        We are starting to suggest bids that match what the opponent is suggesting
            and are within our min max values
        """
        bidScores = []

        for bid in allBids:
            score = self._calculateBidScore(bid)
            utility = self._utilspace.getUtility(bid) if self._utilspace else 0
            combinedScore = Decimal(score) + utility  # Combine the scores
            bidScores.append((bid, combinedScore))

        # Sort the bids by the combined score in descending order
        bidScores.sort(key=lambda x: x[1], reverse=True)

        # Return the top 5 bids
        return [bid for bid, score in bidScores[:5]]

    def _calculateBidScore(self, bid: Bid) -> int:
        """
        We are calculating an index for each value in the AvailbleBids based on the number
            of times it was suggested by the opponent
        high index -> high preference
        low index -> low preference
        """
        score = 0
        for issue, option in bid.getIssueValues().items():
            score += self._optionIndex[issue][option]
        return score

    def _computeMinMax(self, time: float):
        """
        Calculating the limits of our suggestions
        """
        range = self._bidutils.getRange()
        self._maxUtil = range.getMax()

        if 0 <= time < 3:
            self._minUtil = Decimal(0.85 + (time / 30))
        else:
            if (self._counter % 25) == 0:
                self._updateE()

            #self._minUtil = Decimal(self._e - (math.pow((time - 6.1), 3) / (self._e * 1.2 * 150)))
            self._minUtil = self._getUtilityGoal(time)

    def _getUtilityGoal(self, time: float) -> Decimal:
        """
        Changed this function to return the minUtil calculated based on the e value
        """
        util_2 = Decimal(0.4 + 0.6 * (1 - pow(time/10, 2)))

        util = Decimal(self._e - (math.pow((time - 6.1), 3) / (self._e * 1.2 * 150)))

        # bound the value according to [minUtil, maxUtil] in case ft1 is out of the desired range [0,1]
        if util > 0.9 and util_2 < 0.95:
            util = util_2

        if util > self._maxUtil:
            util = self._maxUtil

        return util

    def _evaluate_opponent(self):
        """
        Opponent modelling based on the deviation of their suggestions
        We start evaluating the opponent after the first 3 seconds
        Using standard deviation to calculate the variation of the suggested option from
            the total available options for each issue in the domain
        """
        # Calculation how exploring the agent is, from the different bids he gives us
        opponent_score = 0
        total_num_of_options = 0
        num_issues = len(self._optionIndex)

        for issue, options in self._optionCounts.items():
            indices = list(options.values())
            num_options = len(options)
            total_num_of_options += num_options
            if indices and num_options > 0:
                # Measure the spread of the indices, using standard deviation
                std_dev = np.std(indices)
                opponent_score += std_dev / self._counter

        # Normalize the exploration score by the number of issues
        if num_issues > 0:
            opponent_score /= num_issues

        return opponent_score


