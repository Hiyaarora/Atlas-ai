# Streaming Replication

## How the primary feeds a standby

A standby server connects to the primary and requests the write-ahead log from
a specific position. The primary then streams every record it produces as soon
as it is flushed to disk. Because the standby replays those records in order,
it holds a byte-identical copy of the data directory rather than a logical
reconstruction of the rows. This is why a physical standby cannot replicate a
subset of tables: it is copying the storage layer, not the schema.

The connection is a normal client connection made with the replication
attribute set. It authenticates through the same host-based rules as any other
session, which is a frequent source of confusion for operators who add a user
to the configuration and forget that replication connections are matched by a
separate entry.

## Taking the initial copy

Before a standby can follow anything it needs a base copy of the data
directory taken while the primary is running. The backup tool brackets the
copy with markers that record the log position at which it began, so the
standby knows where to start replaying. Copying the directory with an ordinary
file utility and no such bracketing produces a corrupt standby, because files
copied early and files copied late reflect different moments in time.

On a large database the copy dominates the time to build a standby. Taking it
from an existing standby rather than the primary spares the primary the read
load, at the cost of the copy being as stale as that standby.

## Retention and the disappearing segment problem

The primary recycles write-ahead log files once they are no longer needed for
crash recovery. If a standby falls behind far enough that the segment it wants
has already been recycled, the connection breaks and cannot resume. The
standby must then be rebuilt from a fresh base backup, which on a large
database can take hours.

Two mechanisms prevent this. A replication slot records the oldest position a
consumer still requires, and the primary refuses to recycle anything past that
point. This is reliable but dangerous in the opposite direction: a slot
belonging to a standby that never comes back will pin the log indefinitely
until the disk fills. The alternative, keeping a fixed number of spare
segments, bounds disk usage but offers no guarantee.

Operators are advised to use slots together with an upper bound on how much
the slot may retain, so a dead consumer degrades into a broken standby rather
than an outage on the primary.

## Archiving as a second line of defence

Independently of streaming, the primary can copy each completed segment to
external storage. A standby that has fallen too far behind for streaming can
then fetch the missing segments from the archive and catch up without a
rebuild. The archive is also what makes recovery to an arbitrary past moment
possible, since it retains history that the live server has long recycled.

The archive command must be genuinely durable before it reports success. A
command that writes to a local directory later swept by a cleanup job produces
an archive that appears healthy and is missing exactly the segments needed
during a real recovery.

## Synchronous and asynchronous commit

By default a transaction commits on the primary without waiting for any
standby. Throughput is high and a failover can lose the most recent
transactions. Configuring synchronous commit makes the primary wait for one or
more standbys to confirm receipt before it acknowledges the client. Durability
improves and latency rises by at least one network round trip.

The subtle failure mode is that a synchronous configuration naming a standby
that is offline will block every commit until an administrator intervenes.
Naming multiple candidates, any one of which may acknowledge, avoids turning a
durability feature into a single point of failure.

## Degrees of synchronous confirmation

Confirmation can be required at several depths. The weakest useful setting
waits only for the standby to have received the record into memory. A stronger
one waits for it to be flushed to the standby's disk, which survives the
standby crashing. The strongest waits for the record to be replayed, meaning a
query on the standby immediately afterwards will observe the transaction.

Each step adds latency, and the strongest is rarely appropriate for
transactional workloads because replay can stall behind an expensive
statement, converting a read-side delay into a write-side one.

## Measuring how far behind a standby is

Lag is reported in two different currencies and confusing them leads to wrong
conclusions. Byte lag is the difference between the newest position on the
primary and the position the standby has replayed. Time lag is how old the
last replayed transaction is. A standby that is idle because the primary is
idle shows growing time lag and zero byte lag, which alarms people who are
watching the wrong number.

During a long-running replay of a bulk operation the opposite happens: byte
lag looks small because the standby has received everything, while the replay
position trails because a single record is expensive to apply.

## Why replay falls behind

Replay is single-threaded in the classic design, while the primary generated
the same work across many concurrent sessions. A write-heavy primary can
therefore outrun a standby that is not otherwise loaded, and the gap widens
under exactly the conditions where a fresh standby is most wanted.

Replay also pauses when a record would remove a row that a query running on
the standby still needs to see. The server must choose between delaying replay
and cancelling the query, and both behaviours surprise people the first time
they encounter them.

## Query conflicts on a read replica

A standby serving read traffic can have its queries cancelled to let replay
proceed. Long analytical queries are the usual casualty, because the longer a
query runs the more likely it is that the primary has since removed something
it depends on. Raising the tolerance for delayed replay reduces cancellations
and increases lag, which is the same tradeoff seen from the other side.

An alternative is to have the standby report its oldest running query back to
the primary, so the primary retains the rows in question. This eliminates
cancellations and introduces the risk that an abandoned query on a replica
causes unbounded growth on the primary.

## Promotion

Promoting a standby ends recovery and makes it writable. The former primary
cannot simply be pointed at the new one, because both may have produced
records at the same position, a situation known as a split timeline. A
resynchronisation tool compares the two histories, finds the point at which
they diverged, and rewinds the older server to that point so it can follow
again. Without such a tool the only safe recovery is a full base backup.

A promotion also increments the timeline identifier, which is how every other
server in the cluster recognises that history has forked and refuses to follow
the wrong branch.

## Choosing which standby to promote

When several standbys exist, the one furthest ahead should be promoted, since
any position it lacks is lost. Comparing them requires reading the replay
position from each, which an automated failover system does before deciding.
Promoting an arbitrary standby to save a few seconds discards whatever the
others had received and it did not.

Automated failover must also guard against promoting while the old primary is
merely unreachable rather than dead, because two writable servers accepting
traffic simultaneously produces divergence that no tool can merge.

## Cascading

A standby may itself feed further standbys. This reduces the network burden on
the primary in geographically distributed deployments, at the cost of an extra
replay hop of latency for the leaf servers. A cascading standby continues to
serve its own downstreams while it is being promoted, which makes it useful as
a regional hub.

Cascaded servers see the timeline change propagate through their upstream, so
a promotion at the root is visible throughout the tree without reconfiguring
each leaf individually.

## Logical replication and where it differs

Logical replication ships row changes rather than storage blocks. Because it
operates on rows it can replicate a subset of tables, feed a server running a
different major version, and target a schema that is not identical. Those
capabilities are the reason to choose it, and they come with constraints that
physical replication does not have.

Changes to the schema are not replicated. A column added on the publisher must
be added on the subscriber separately, and doing so in the wrong order breaks
the stream. Sequences are likewise not synchronised, which matters at the
moment a subscriber is promoted to serve writes.

## Conflicts in logical replication

A subscriber applies incoming changes as ordinary writes, so anything that
would break a local constraint stops the stream. A row inserted directly on
the subscriber that later collides with a replicated insert halts replication
until an operator resolves it. The stream does not skip the offending change
on its own, because silently dropping a row would leave the two databases
permanently and invisibly divergent.

The practical guidance is that a subscriber's replicated tables should be
treated as read-only, and any exception to that should be deliberate.

## Monitoring what actually matters

Three signals are worth alerting on. Whether each expected standby is
connected at all, since a disconnected standby is a silent loss of redundancy.
How far behind each one is, measured in bytes for a threshold and in time for
a human-readable summary. And how much log each slot is retaining, because
that is the metric that predicts a full disk on the primary.

Alerting on lag alone is a common mistake. A standby that has disconnected
entirely stops reporting lag, so a naive check sees no lag and concludes
everything is healthy.

## Capacity and network considerations

Replication traffic is roughly proportional to the volume of write-ahead log
generated, which is not the same as the volume of data changed. An update
touching one column of a wide row may log the entire row, and index
maintenance adds further records. Estimating bandwidth from the size of the
data set rather than from measured log generation understates it consistently.

Compressing the stream helps on constrained links and costs processor time on
both ends. Whether that trade is worth it depends on which resource is scarce,
which is a measurement rather than a rule.
